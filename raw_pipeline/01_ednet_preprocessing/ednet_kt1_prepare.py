#!/usr/bin/env python3
"""Robust, resumable EdNet-KT1 directory processor.

Reads the extracted KT1 directory (one CSV per learner), joins the official
EdNet questions metadata from EdNet-Contents.zip or questions.csv, validates
and flags records, then writes atomic sharded outputs.

Primary output format: Parquet (requires pyarrow). A csv-gzip backend is kept
for validation and portability.
"""
from __future__ import annotations

import argparse
import csv
import collections
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

REQUIRED_KT1_COLUMNS = (
    "timestamp",
    "solving_id",
    "question_id",
    "user_answer",
    "elapsed_time",
)
REQUIRED_QUESTION_COLUMNS = (
    "question_id",
    "bundle_id",
    "explanation_id",
    "correct_answer",
    "part",
    "tags",
    "deployed_at",
)
USER_FILE_RE = re.compile(r"^u(\d+)\.csv$", re.IGNORECASE)
QUESTION_ID_RE = re.compile(r"^q\d+$")
VALID_ANSWERS = frozenset({"a", "b", "c", "d"})

# Quality bit mask. Values are stable and must not be renumbered.
Q_UNKNOWN_QUESTION = 1 << 0
Q_MISSING_CORRECT_ANSWER = 1 << 1
Q_MISSING_SKILL_TAG = 1 << 2
Q_NON_MONOTONIC_INPUT = 1 << 3
Q_DUPLICATE_EVENT = 1 << 4
Q_INVALID_RESPONSE = 1 << 5
Q_NEGATIVE_ELAPSED = 1 << 6
Q_EXTREME_ELAPSED = 1 << 7
Q_QUESTION_BEFORE_DEPLOYMENT = 1 << 8
Q_INVALID_TIMESTAMP = 1 << 9
Q_INVALID_SOLVING_ID = 1 << 10
Q_INVALID_ELAPSED = 1 << 11

QUALITY_FLAG_DEFINITIONS = {
    str(Q_UNKNOWN_QUESTION): "UNKNOWN_QUESTION",
    str(Q_MISSING_CORRECT_ANSWER): "MISSING_CORRECT_ANSWER",
    str(Q_MISSING_SKILL_TAG): "MISSING_SKILL_TAG",
    str(Q_NON_MONOTONIC_INPUT): "NON_MONOTONIC_INPUT",
    str(Q_DUPLICATE_EVENT): "DUPLICATE_EVENT",
    str(Q_INVALID_RESPONSE): "INVALID_RESPONSE",
    str(Q_NEGATIVE_ELAPSED): "NEGATIVE_ELAPSED_TIME",
    str(Q_EXTREME_ELAPSED): "EXTREME_ELAPSED_TIME",
    str(Q_QUESTION_BEFORE_DEPLOYMENT): "QUESTION_BEFORE_DEPLOYMENT",
    str(Q_INVALID_TIMESTAMP): "INVALID_TIMESTAMP",
    str(Q_INVALID_SOLVING_ID): "INVALID_SOLVING_ID",
    str(Q_INVALID_ELAPSED): "INVALID_ELAPSED_TIME",
}

INTERACTION_COLUMNS = (
    "learner_id",
    "sequence_index",
    "timestamp_ms",
    "solving_id",
    "question_id",
    "bundle_id",
    "user_answer",
    "correct_answer",
    "is_correct",
    "elapsed_time_ms",
    "part",
    "skill_ids",
    "skill_count",
    "primary_skill_id",
    "deployed_at_ms",
    "source_file",
    "source_row",
    "quality_flags",
)

LEARNER_COLUMNS = (
    "learner_id",
    "source_file",
    "raw_rows",
    "output_rows",
    "valid_correctness_rows",
    "correct_rows",
    "first_timestamp_ms",
    "last_timestamp_ms",
    "non_monotonic_count",
    "duplicate_count",
    "unknown_question_count",
    "invalid_response_count",
    "missing_skill_count",
    "negative_elapsed_count",
    "extreme_elapsed_count",
    "question_before_deployment_count",
    "file_error",
)

QUESTION_OUTPUT_COLUMNS = (
    "question_id",
    "bundle_id",
    "explanation_id",
    "correct_answer",
    "part",
    "skill_ids",
    "skill_count",
    "primary_skill_id",
    "deployed_at_ms",
)


class ProcessingError(RuntimeError):
    """Fatal configuration, schema or integrity error."""


@dataclass(frozen=True)
class QuestionMeta:
    question_id: str
    bundle_id: str | None
    explanation_id: str | None
    correct_answer: str | None
    part: int | None
    skill_ids: str
    skill_count: int
    primary_skill_id: int | None
    deployed_at_ms: int | None


@dataclass
class LearnerResult:
    interactions: list[tuple[Any, ...]]
    learner_summary: tuple[Any, ...]
    error_record: dict[str, Any] | None


@dataclass
class ShardStats:
    shard_id: int
    first_file: str
    last_file: str
    input_files: int
    interaction_rows: int
    learner_rows: int
    file_errors: int
    started_at_utc: str
    completed_at_utc: str
    elapsed_seconds: float
    interaction_output: str
    learner_output: str
    error_output: str | None
    interaction_sha256: str
    learner_sha256: str
    error_sha256: str | None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding=encoding, newline="") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def parse_tags(raw: str | None) -> tuple[str, int, int | None]:
    if raw is None:
        return "", 0, None
    text = str(raw).strip()
    if text in {"", "-1", "nan", "None", "null"}:
        return "", 0, None
    parsed: list[int] = []
    for token in text.split(";"):
        token = token.strip()
        if not token:
            continue
        if not re.fullmatch(r"-?\d+", token):
            continue
        value = int(token)
        if value < 0:
            continue
        parsed.append(value)
    # Preserve first occurrence order and remove duplicates.
    parsed = list(dict.fromkeys(parsed))
    normalized = ";".join(str(x) for x in parsed)
    primary = parsed[0] if len(parsed) == 1 else None
    return normalized, len(parsed), primary


def normalize_answer(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def learner_id_from_path(path: Path) -> int:
    match = USER_FILE_RE.match(path.name)
    if not match:
        raise ProcessingError(f"Invalid KT1 learner filename: {path.name!r}; expected u<integer>.csv")
    return int(match.group(1))


def locate_questions_member(zf: zipfile.ZipFile) -> str:
    candidates = [
        name for name in zf.namelist()
        if not name.endswith("/") and Path(name).name.lower() == "questions.csv"
    ]
    candidates = [name for name in candidates if "__MACOSX" not in name]
    if len(candidates) != 1:
        raise ProcessingError(
            f"Expected exactly one questions.csv in contents ZIP, found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def _read_questions_rows(contents_source: Path) -> tuple[list[dict[str, str]], str]:
    if not contents_source.exists():
        raise ProcessingError(f"Contents source does not exist: {contents_source}")
    if contents_source.is_dir():
        candidates = list(contents_source.rglob("questions.csv"))
        candidates = [p for p in candidates if "__MACOSX" not in p.parts]
        if len(candidates) != 1:
            raise ProcessingError(
                f"Expected exactly one questions.csv under {contents_source}, found {len(candidates)}"
            )
        qpath = candidates[0]
        with qpath.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        return rows, sha256_file(qpath)
    if contents_source.suffix.lower() == ".zip":
        with zipfile.ZipFile(contents_source, "r") as zf:
            member = locate_questions_member(zf)
            raw = zf.read(member)
        digest = hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8-sig")
        reader = csv.DictReader(text.splitlines())
        return list(reader), digest
    if contents_source.name.lower() == "questions.csv":
        with contents_source.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        return rows, sha256_file(contents_source)
    raise ProcessingError(
        "--contents must point to EdNet-Contents.zip, an extracted contents directory, or questions.csv"
    )


def load_question_metadata(contents_source: Path) -> tuple[dict[str, QuestionMeta], dict[str, Any]]:
    rows, questions_sha256 = _read_questions_rows(contents_source)
    if not rows:
        raise ProcessingError("questions.csv is empty")
    actual_columns = set(rows[0].keys())
    missing = [c for c in REQUIRED_QUESTION_COLUMNS if c not in actual_columns]
    if missing:
        raise ProcessingError(f"questions.csv missing required columns: {missing}")

    metadata: dict[str, QuestionMeta] = {}
    duplicate_ids: list[str] = []
    invalid_ids: list[str] = []
    invalid_answers: list[str] = []
    missing_skill_rows = 0
    missing_deployed_rows = 0

    for row_number, row in enumerate(rows, start=2):
        qid = (row.get("question_id") or "").strip()
        if not QUESTION_ID_RE.fullmatch(qid):
            invalid_ids.append(f"row {row_number}: {qid!r}")
            continue
        if qid in metadata:
            duplicate_ids.append(qid)
            continue
        answer = normalize_answer(row.get("correct_answer"))
        if answer not in VALID_ANSWERS:
            invalid_answers.append(f"row {row_number}: {qid}={answer!r}")
            answer = None
        skill_ids, skill_count, primary_skill_id = parse_tags(row.get("tags"))
        if skill_count == 0:
            missing_skill_rows += 1
        deployed = parse_int(row.get("deployed_at"))
        if deployed is None or deployed < 0:
            deployed = None
            missing_deployed_rows += 1
        part = parse_int(row.get("part"))
        metadata[qid] = QuestionMeta(
            question_id=qid,
            bundle_id=(row.get("bundle_id") or "").strip() or None,
            explanation_id=(row.get("explanation_id") or "").strip() or None,
            correct_answer=answer,
            part=part,
            skill_ids=skill_ids,
            skill_count=skill_count,
            primary_skill_id=primary_skill_id,
            deployed_at_ms=deployed,
        )

    if duplicate_ids:
        raise ProcessingError(f"Duplicate question_id values in questions.csv: {duplicate_ids[:10]}")
    if invalid_ids:
        raise ProcessingError(f"Invalid question_id values in questions.csv: {invalid_ids[:10]}")
    if invalid_answers:
        raise ProcessingError(f"Invalid correct_answer values in questions.csv: {invalid_answers[:10]}")
    if not metadata:
        raise ProcessingError("No valid question metadata could be loaded")

    summary = {
        "questions_sha256": questions_sha256,
        "question_rows": len(rows),
        "unique_questions": len(metadata),
        "missing_skill_rows": missing_skill_rows,
        "missing_deployed_rows": missing_deployed_rows,
        "required_columns": list(REQUIRED_QUESTION_COLUMNS),
    }
    return metadata, summary


def enumerate_kt1_files(kt1_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    if not kt1_dir.exists() or not kt1_dir.is_dir():
        raise ProcessingError(f"KT1 directory does not exist or is not a directory: {kt1_dir}")
    files: list[tuple[int, Path, int]] = []
    unexpected_csv: list[str] = []
    total_bytes = 0
    name_digest = hashlib.sha256()

    with os.scandir(kt1_dir) as iterator:
        for entry in iterator:
            if not entry.is_file():
                continue
            if not entry.name.lower().endswith(".csv"):
                continue
            match = USER_FILE_RE.match(entry.name)
            if not match:
                unexpected_csv.append(entry.name)
                continue
            learner_num = int(match.group(1))
            path = Path(entry.path)
            stat = entry.stat()
            total_bytes += stat.st_size
            files.append((learner_num, path, stat.st_size))

    if unexpected_csv:
        raise ProcessingError(
            f"Unexpected CSV filenames in KT1 directory (first 20): {unexpected_csv[:20]}"
        )
    if not files:
        raise ProcessingError(f"No learner CSV files matching u<integer>.csv found in {kt1_dir}")

    files.sort(key=lambda pair: (pair[0], pair[1].name.lower()))
    ids = [item[0] for item in files]
    id_counts = collections.Counter(ids)
    duplicates = [learner_id for learner_id, count in id_counts.items() if count > 1]
    if duplicates:
        raise ProcessingError(f"Duplicate learner numeric IDs detected: {sorted(duplicates)[:20]}")

    sorted_paths = [item[1] for item in files]
    for learner_num, path, file_size in files:
        name_digest.update(f"{learner_num}\t{path.name}\t{file_size}\n".encode("utf-8"))

    summary = {
        "kt1_directory": str(kt1_dir.resolve()),
        "learner_file_count": len(sorted_paths),
        "first_file": sorted_paths[0].name,
        "last_file": sorted_paths[-1].name,
        "minimum_learner_id": min(ids),
        "maximum_learner_id": max(ids),
        "total_input_bytes": total_bytes,
        "directory_fingerprint": name_digest.hexdigest(),
    }
    return sorted_paths, summary


def validate_header(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise ProcessingError(f"Empty learner CSV file: {path}")
    normalized = tuple(col.strip() for col in header)
    missing = [c for c in REQUIRED_KT1_COLUMNS if c not in normalized]
    if missing:
        raise ProcessingError(f"{path.name} missing required KT1 columns {missing}; header={normalized}")
    return normalized


def parse_learner_file(
    path: Path,
    question_meta: Mapping[str, QuestionMeta],
    extreme_elapsed_ms: int,
) -> LearnerResult:
    learner_id = learner_id_from_path(path)
    source_file = path.name
    interactions: list[tuple[Any, ...]] = []
    raw_rows: list[dict[str, Any]] = []
    error_record: dict[str, Any] | None = None

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise ProcessingError("missing CSV header")
            normalized = [col.strip() for col in reader.fieldnames]
            missing = [c for c in REQUIRED_KT1_COLUMNS if c not in normalized]
            if missing:
                raise ProcessingError(f"missing required columns {missing}; header={normalized}")
            # DictReader uses original names. Standard EdNet columns contain no spaces.
            for source_row, row in enumerate(reader, start=2):
                raw_rows.append({
                    "source_row": source_row,
                    "timestamp_raw": row.get("timestamp"),
                    "solving_id_raw": row.get("solving_id"),
                    "question_id_raw": row.get("question_id"),
                    "user_answer_raw": row.get("user_answer"),
                    "elapsed_time_raw": row.get("elapsed_time"),
                })
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        error_record = {
            "learner_id": learner_id,
            "source_file": source_file,
            "stage": "read",
            "error": message,
            "traceback": traceback.format_exc(limit=5),
        }
        summary = (
            learner_id, source_file, 0, 0, 0, 0, None, None,
            0, 0, 0, 0, 0, 0, 0, 0, message,
        )
        return LearnerResult([], summary, error_record)

    non_monotonic_source_rows: set[int] = set()
    previous_valid_timestamp: int | None = None
    parsed_for_sort: list[tuple[int, int, dict[str, Any]]] = []
    invalid_timestamp_rows: list[dict[str, Any]] = []

    for raw in raw_rows:
        timestamp = parse_int(raw["timestamp_raw"])
        if timestamp is None or timestamp < 0:
            invalid_timestamp_rows.append(raw)
            continue
        if previous_valid_timestamp is not None and timestamp < previous_valid_timestamp:
            non_monotonic_source_rows.add(raw["source_row"])
        previous_valid_timestamp = timestamp
        parsed_for_sort.append((timestamp, raw["source_row"], raw))

    # Valid timestamps are sorted stably. Invalid timestamp rows remain at the end
    # and are retained with a quality flag and null timestamp.
    parsed_for_sort.sort(key=lambda item: (item[0], item[1]))
    ordered: list[tuple[int | None, dict[str, Any]]] = [
        (timestamp, raw) for timestamp, _, raw in parsed_for_sort
    ] + [(None, raw) for raw in invalid_timestamp_rows]

    duplicate_keys: set[tuple[Any, ...]] = set()
    seen_keys: set[tuple[Any, ...]] = set()

    valid_correctness_rows = 0
    correct_rows = 0
    unknown_question_count = 0
    invalid_response_count = 0
    missing_skill_count = 0
    negative_elapsed_count = 0
    extreme_elapsed_count = 0
    question_before_deployment_count = 0
    duplicate_count = 0

    valid_timestamps: list[int] = []

    for sequence_index, (timestamp, raw) in enumerate(ordered):
        flags = 0
        if raw["source_row"] in non_monotonic_source_rows:
            flags |= Q_NON_MONOTONIC_INPUT
        if timestamp is None:
            flags |= Q_INVALID_TIMESTAMP
        else:
            valid_timestamps.append(timestamp)

        solving_id = parse_int(raw["solving_id_raw"])
        if solving_id is None:
            flags |= Q_INVALID_SOLVING_ID

        qid = (str(raw["question_id_raw"]).strip() if raw["question_id_raw"] is not None else "")
        user_answer = normalize_answer(raw["user_answer_raw"])
        elapsed = parse_int(raw["elapsed_time_raw"])
        if elapsed is None and str(raw["elapsed_time_raw"] or "").strip() != "":
            flags |= Q_INVALID_ELAPSED
        if elapsed is not None and elapsed < 0:
            flags |= Q_NEGATIVE_ELAPSED
            negative_elapsed_count += 1
        if elapsed is not None and elapsed > extreme_elapsed_ms:
            flags |= Q_EXTREME_ELAPSED
            extreme_elapsed_count += 1

        meta = question_meta.get(qid)
        if meta is None:
            flags |= Q_UNKNOWN_QUESTION | Q_MISSING_CORRECT_ANSWER | Q_MISSING_SKILL_TAG
            unknown_question_count += 1
            missing_skill_count += 1
            bundle_id = None
            correct_answer = None
            is_correct = None
            part = None
            skill_ids = ""
            skill_count = 0
            primary_skill_id = None
            deployed_at = None
        else:
            bundle_id = meta.bundle_id
            correct_answer = meta.correct_answer
            part = meta.part
            skill_ids = meta.skill_ids
            skill_count = meta.skill_count
            primary_skill_id = meta.primary_skill_id
            deployed_at = meta.deployed_at_ms
            if correct_answer is None:
                flags |= Q_MISSING_CORRECT_ANSWER
                is_correct = None
            elif user_answer not in VALID_ANSWERS:
                flags |= Q_INVALID_RESPONSE
                invalid_response_count += 1
                is_correct = None
            else:
                is_correct = user_answer == correct_answer
                valid_correctness_rows += 1
                correct_rows += int(is_correct)
            if skill_count == 0:
                flags |= Q_MISSING_SKILL_TAG
                missing_skill_count += 1
            if timestamp is not None and deployed_at is not None and timestamp < deployed_at:
                flags |= Q_QUESTION_BEFORE_DEPLOYMENT
                question_before_deployment_count += 1

        if meta is None and user_answer not in VALID_ANSWERS:
            flags |= Q_INVALID_RESPONSE
            invalid_response_count += 1

        duplicate_key = (
            timestamp,
            solving_id,
            qid,
            user_answer,
            elapsed,
        )
        if duplicate_key in seen_keys:
            flags |= Q_DUPLICATE_EVENT
            duplicate_keys.add(duplicate_key)
            duplicate_count += 1
        else:
            seen_keys.add(duplicate_key)

        interactions.append((
            learner_id,
            sequence_index,
            timestamp,
            solving_id,
            qid or None,
            bundle_id,
            user_answer,
            correct_answer,
            is_correct,
            elapsed,
            part,
            skill_ids,
            skill_count,
            primary_skill_id,
            deployed_at,
            source_file,
            raw["source_row"],
            flags,
        ))

    learner_summary = (
        learner_id,
        source_file,
        len(raw_rows),
        len(interactions),
        valid_correctness_rows,
        correct_rows,
        min(valid_timestamps) if valid_timestamps else None,
        max(valid_timestamps) if valid_timestamps else None,
        len(non_monotonic_source_rows),
        duplicate_count,
        unknown_question_count,
        invalid_response_count,
        missing_skill_count,
        negative_elapsed_count,
        extreme_elapsed_count,
        question_before_deployment_count,
        None,
    )
    return LearnerResult(interactions, learner_summary, error_record)


def require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        raise ProcessingError(
            "Parquet output requires pyarrow. Run the supplied PowerShell launcher, "
            "or install exactly: python -m pip install pyarrow==25.0.0"
        ) from exc
    return pa, pq


def parquet_schemas() -> tuple[Any, Any, Any]:
    pa, _ = require_pyarrow()
    interaction_schema = pa.schema([
        ("learner_id", pa.int32()),
        ("sequence_index", pa.int32()),
        ("timestamp_ms", pa.int64()),
        ("solving_id", pa.int64()),
        ("question_id", pa.string()),
        ("bundle_id", pa.string()),
        ("user_answer", pa.string()),
        ("correct_answer", pa.string()),
        ("is_correct", pa.bool_()),
        ("elapsed_time_ms", pa.int64()),
        ("part", pa.int8()),
        ("skill_ids", pa.string()),
        ("skill_count", pa.int16()),
        ("primary_skill_id", pa.int32()),
        ("deployed_at_ms", pa.int64()),
        ("source_file", pa.string()),
        ("source_row", pa.int32()),
        ("quality_flags", pa.uint16()),
    ])
    learner_schema = pa.schema([
        ("learner_id", pa.int32()),
        ("source_file", pa.string()),
        ("raw_rows", pa.int32()),
        ("output_rows", pa.int32()),
        ("valid_correctness_rows", pa.int32()),
        ("correct_rows", pa.int32()),
        ("first_timestamp_ms", pa.int64()),
        ("last_timestamp_ms", pa.int64()),
        ("non_monotonic_count", pa.int32()),
        ("duplicate_count", pa.int32()),
        ("unknown_question_count", pa.int32()),
        ("invalid_response_count", pa.int32()),
        ("missing_skill_count", pa.int32()),
        ("negative_elapsed_count", pa.int32()),
        ("extreme_elapsed_count", pa.int32()),
        ("question_before_deployment_count", pa.int32()),
        ("file_error", pa.string()),
    ])
    question_schema = pa.schema([
        ("question_id", pa.string()),
        ("bundle_id", pa.string()),
        ("explanation_id", pa.string()),
        ("correct_answer", pa.string()),
        ("part", pa.int8()),
        ("skill_ids", pa.string()),
        ("skill_count", pa.int16()),
        ("primary_skill_id", pa.int32()),
        ("deployed_at_ms", pa.int64()),
    ])
    return interaction_schema, learner_schema, question_schema


def write_parquet_atomic(
    output_path: Path,
    columns: Sequence[str],
    rows: Sequence[tuple[Any, ...]],
    schema: Any,
) -> None:
    pa, pq = require_pyarrow()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".tmp")
    data = {name: [row[idx] for row in rows] for idx, name in enumerate(columns)}
    table = pa.Table.from_pydict(data, schema=schema)
    pq.write_table(
        table,
        tmp,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        write_statistics=True,
        row_group_size=131_072,
        data_page_size=1_048_576,
    )
    # Read-back verification before atomic publication.
    check = pq.read_metadata(tmp)
    if check.num_rows != len(rows):
        tmp.unlink(missing_ok=True)
        raise ProcessingError(
            f"Parquet verification failed for {output_path}: expected {len(rows)} rows, got {check.num_rows}"
        )
    os.replace(tmp, output_path)


def write_csv_gzip_atomic(
    output_path: Path,
    columns: Sequence[str],
    rows: Sequence[tuple[Any, ...]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", newline="", compresslevel=6) as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)
    # Verify readability and row count before publication.
    count = 0
    with gzip.open(tmp, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if tuple(header or ()) != tuple(columns):
            tmp.unlink(missing_ok=True)
            raise ProcessingError(f"CSV.GZ header verification failed for {output_path}")
        for _ in reader:
            count += 1
    if count != len(rows):
        tmp.unlink(missing_ok=True)
        raise ProcessingError(
            f"CSV.GZ verification failed for {output_path}: expected {len(rows)} rows, got {count}"
        )
    os.replace(tmp, output_path)


def output_extension(output_format: str) -> str:
    return ".parquet" if output_format == "parquet" else ".csv.gz"


def write_table_atomic(
    output_path: Path,
    columns: Sequence[str],
    rows: Sequence[tuple[Any, ...]],
    output_format: str,
    schema: Any | None,
) -> None:
    if output_format == "parquet":
        write_parquet_atomic(output_path, columns, rows, schema)
    elif output_format == "csv-gzip":
        write_csv_gzip_atomic(output_path, columns, rows)
    else:
        raise ProcessingError(f"Unsupported output format: {output_format}")


def write_error_records_atomic(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    # Verify each line is valid JSON.
    count = 0
    with gzip.open(tmp, "rt", encoding="utf-8") as fh:
        for line in fh:
            json.loads(line)
            count += 1
    if count != len(records):
        tmp.unlink(missing_ok=True)
        raise ProcessingError(f"Error-log verification failed for {path}")
    os.replace(tmp, path)


def write_question_metadata(
    output_dir: Path,
    metadata: Mapping[str, QuestionMeta],
    output_format: str,
) -> Path:
    extension = output_extension(output_format)
    output_path = output_dir / "metadata" / f"questions{extension}"
    if output_path.exists():
        return output_path
    rows = [
        (
            q.question_id,
            q.bundle_id,
            q.explanation_id,
            q.correct_answer,
            q.part,
            q.skill_ids,
            q.skill_count,
            q.primary_skill_id,
            q.deployed_at_ms,
        )
        for q in sorted(metadata.values(), key=lambda x: int(x.question_id[1:]))
    ]
    q_schema = parquet_schemas()[2] if output_format == "parquet" else None
    write_table_atomic(output_path, QUESTION_OUTPUT_COLUMNS, rows, output_format, q_schema)
    return output_path


def load_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "created_at_utc": utc_now_iso(),
            "completed_shards": {},
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ProcessingError(f"Cannot read progress file {path}: {exc}") from exc


def check_resume_compatibility(progress: Mapping[str, Any], run_identity: Mapping[str, Any]) -> None:
    existing = progress.get("run_identity")
    if existing is None:
        return
    if existing != run_identity:
        raise ProcessingError(
            "Output directory contains progress from a different input/configuration. "
            "Use a new output directory or restore the original parameters.\n"
            f"Existing identity: {json.dumps(existing, ensure_ascii=False, sort_keys=True)}\n"
            f"Current identity:  {json.dumps(run_identity, ensure_ascii=False, sort_keys=True)}"
        )


def evenly_spaced_sample(items: Sequence[Path], count: int) -> list[Path]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    indices = sorted({round(i * (len(items) - 1) / (count - 1)) for i in range(count)})
    return [items[i] for i in indices]


def validate_real_sample(
    sample_files: Sequence[Path],
    question_meta: Mapping[str, QuestionMeta],
    extreme_elapsed_ms: int,
) -> dict[str, Any]:
    if not sample_files:
        raise ProcessingError("No files selected for real-data preflight sample")
    results: list[LearnerResult] = []
    for path in sample_files:
        validate_header(path)
        result = parse_learner_file(path, question_meta, extreme_elapsed_ms)
        if result.error_record is not None:
            raise ProcessingError(
                f"Preflight parse failed for {path.name}: {result.error_record['error']}"
            )
        results.append(result)
    interactions = sum(len(r.interactions) for r in results)
    unknown = sum(r.learner_summary[10] for r in results)
    return {
        "sample_files": len(sample_files),
        "sample_interactions": interactions,
        "sample_unknown_questions": unknown,
        "sample_file_names": [p.name for p in sample_files],
    }


def run_internal_self_tests(output_format: str) -> dict[str, Any]:
    """Exercise parsing, metadata join, flags, atomic writer and read-back."""
    with tempfile.TemporaryDirectory(prefix="ednet_selftest_") as tmp_name:
        tmp = Path(tmp_name)
        contents = tmp / "questions.csv"
        contents.write_text(
            "question_id,bundle_id,explanation_id,correct_answer,part,tags,deployed_at\n"
            "q1,b1,e1,a,1,10,1000\n"
            "q2,b2,e2,b,2,-1,-1\n",
            encoding="utf-8",
        )
        meta, _ = load_question_metadata(contents)
        kt1 = tmp / "KT1"
        kt1.mkdir()
        (kt1 / "u1.csv").write_text(
            "timestamp,solving_id,question_id,user_answer,elapsed_time\n"
            "2000,1,q1,a,1000\n"
            "1500,1,q2,c,-5\n"
            "1500,1,q2,c,-5\n"
            "3000,2,q999,z,abc\n",
            encoding="utf-8",
        )
        result = parse_learner_file(kt1 / "u1.csv", meta, extreme_elapsed_ms=2000)
        assert result.error_record is None
        assert len(result.interactions) == 4
        assert result.learner_summary[8] == 1  # one non-monotonic source row
        assert result.learner_summary[9] == 1  # one duplicate
        assert result.learner_summary[10] == 1  # unknown q
        assert result.learner_summary[11] == 1  # invalid response
        assert result.learner_summary[12] >= 3  # q2/q999 missing skills
        assert result.learner_summary[13] == 2  # negative elapsed duplicate included

        extension = output_extension(output_format)
        interaction_path = tmp / f"interactions{extension}"
        learner_path = tmp / f"learners{extension}"
        if output_format == "parquet":
            interaction_schema, learner_schema, _ = parquet_schemas()
        else:
            interaction_schema = learner_schema = None
        write_table_atomic(
            interaction_path,
            INTERACTION_COLUMNS,
            result.interactions,
            output_format,
            interaction_schema,
        )
        write_table_atomic(
            learner_path,
            LEARNER_COLUMNS,
            [result.learner_summary],
            output_format,
            learner_schema,
        )
        assert interaction_path.exists() and interaction_path.stat().st_size > 0
        assert learner_path.exists() and learner_path.stat().st_size > 0
        return {
            "status": "PASS",
            "interaction_rows": len(result.interactions),
            "output_format": output_format,
        }


def estimate_output_space(
    sample_files: Sequence[Path],
    question_meta: Mapping[str, QuestionMeta],
    output_format: str,
    extreme_elapsed_ms: int,
    total_input_bytes: int,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ednet_space_estimate_") as tmp_name:
        tmp = Path(tmp_name)
        results = [parse_learner_file(p, question_meta, extreme_elapsed_ms) for p in sample_files]
        interaction_rows = [row for result in results for row in result.interactions]
        learner_rows = [result.learner_summary for result in results]
        sample_raw_bytes = sum(p.stat().st_size for p in sample_files)
        if sample_raw_bytes <= 0:
            raise ProcessingError("Cannot estimate output size from zero-byte sample")
        extension = output_extension(output_format)
        if output_format == "parquet":
            i_schema, l_schema, _ = parquet_schemas()
        else:
            i_schema = l_schema = None
        ipath = tmp / f"i{extension}"
        lpath = tmp / f"l{extension}"
        write_table_atomic(ipath, INTERACTION_COLUMNS, interaction_rows, output_format, i_schema)
        write_table_atomic(lpath, LEARNER_COLUMNS, learner_rows, output_format, l_schema)
        sample_output_bytes = ipath.stat().st_size + lpath.stat().st_size
        raw_ratio = sample_output_bytes / sample_raw_bytes
        # Safety factor includes shard overhead and sample variability.
        estimated = math.ceil(total_input_bytes * raw_ratio * 1.35)
        return {
            "sample_raw_bytes": sample_raw_bytes,
            "sample_output_bytes": sample_output_bytes,
            "observed_output_to_raw_ratio": raw_ratio,
            "estimated_output_bytes_with_safety_factor": estimated,
        }


def human_bytes(value: int | float) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024.0 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TiB"


def process_shard(
    shard_id: int,
    files: Sequence[Path],
    question_meta: Mapping[str, QuestionMeta],
    output_dir: Path,
    output_format: str,
    extreme_elapsed_ms: int,
    workers: int,
) -> ShardStats:
    started_wall = time.perf_counter()
    started_utc = utc_now_iso()

    if workers == 1:
        results = [parse_learner_file(p, question_meta, extreme_elapsed_ms) for p in files]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ednet") as executor:
            results = list(
                executor.map(
                    lambda p: parse_learner_file(p, question_meta, extreme_elapsed_ms),
                    files,
                )
            )

    interaction_rows = [row for result in results for row in result.interactions]
    learner_rows = [result.learner_summary for result in results]
    error_records = [result.error_record for result in results if result.error_record is not None]

    extension = output_extension(output_format)
    interaction_path = output_dir / "interactions" / f"part-{shard_id:06d}{extension}"
    learner_path = output_dir / "learners" / f"part-{shard_id:06d}{extension}"
    error_path = output_dir / "errors" / f"part-{shard_id:06d}.jsonl.gz"

    if output_format == "parquet":
        interaction_schema, learner_schema, _ = parquet_schemas()
    else:
        interaction_schema = learner_schema = None

    write_table_atomic(
        interaction_path,
        INTERACTION_COLUMNS,
        interaction_rows,
        output_format,
        interaction_schema,
    )
    write_table_atomic(
        learner_path,
        LEARNER_COLUMNS,
        learner_rows,
        output_format,
        learner_schema,
    )
    if error_records:
        write_error_records_atomic(error_path, error_records)
    elif error_path.exists():
        error_path.unlink()

    elapsed = time.perf_counter() - started_wall
    return ShardStats(
        shard_id=shard_id,
        first_file=files[0].name,
        last_file=files[-1].name,
        input_files=len(files),
        interaction_rows=len(interaction_rows),
        learner_rows=len(learner_rows),
        file_errors=len(error_records),
        started_at_utc=started_utc,
        completed_at_utc=utc_now_iso(),
        elapsed_seconds=elapsed,
        interaction_output=str(interaction_path.relative_to(output_dir)),
        learner_output=str(learner_path.relative_to(output_dir)),
        error_output=str(error_path.relative_to(output_dir)) if error_records else None,
        interaction_sha256=sha256_file(interaction_path),
        learner_sha256=sha256_file(learner_path),
        error_sha256=sha256_file(error_path) if error_records else None,
    )


def verify_completed_shard(output_dir: Path, stats: Mapping[str, Any]) -> bool:
    try:
        interaction_path = output_dir / stats["interaction_output"]
        learner_path = output_dir / stats["learner_output"]
        if not interaction_path.exists() or not learner_path.exists():
            return False
        if sha256_file(interaction_path) != stats["interaction_sha256"]:
            return False
        if sha256_file(learner_path) != stats["learner_sha256"]:
            return False
        error_output = stats.get("error_output")
        if error_output:
            error_path = output_dir / error_output
            if not error_path.exists() or sha256_file(error_path) != stats.get("error_sha256"):
                return False
        return True
    except Exception:
        return False


def build_final_manifest(
    output_dir: Path,
    progress: Mapping[str, Any],
    kt1_summary: Mapping[str, Any],
    question_summary: Mapping[str, Any],
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    completed = progress.get("completed_shards", {})
    shard_stats = [completed[key] for key in sorted(completed, key=lambda x: int(x))]
    interaction_rows = sum(int(s["interaction_rows"]) for s in shard_stats)
    learner_rows = sum(int(s["learner_rows"]) for s in shard_stats)
    file_errors = sum(int(s["file_errors"]) for s in shard_stats)
    return {
        "status": "COMPLETED_WITH_FILE_ERRORS" if file_errors else "COMPLETED",
        "completed_at_utc": utc_now_iso(),
        "kt1": dict(kt1_summary),
        "questions": dict(question_summary),
        "parameters": {
            "output_format": args.output_format,
            "files_per_shard": args.files_per_shard,
            "workers": args.workers,
            "extreme_elapsed_ms": args.extreme_elapsed_ms,
        },
        "preflight": dict(preflight),
        "output": {
            "shard_count": len(shard_stats),
            "interaction_rows": interaction_rows,
            "learner_rows": learner_rows,
            "file_errors": file_errors,
            "quality_flag_definitions": QUALITY_FLAG_DEFINITIONS,
        },
        "shards": shard_stats,
    }


def perform_preflight(
    args: argparse.Namespace,
    files: Sequence[Path],
    question_meta: Mapping[str, QuestionMeta],
    kt1_summary: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    if sys.version_info < (3, 10):
        raise ProcessingError(f"Python 3.10+ is required; found {sys.version}")
    if args.output_format == "parquet":
        pa, _ = require_pyarrow()
        pyarrow_version = pa.__version__
    else:
        pyarrow_version = None

    output_dir.mkdir(parents=True, exist_ok=True)
    write_test = output_dir / ".write_test.tmp"
    write_test.write_text("ok", encoding="utf-8")
    write_test.unlink()

    self_test = run_internal_self_tests(args.output_format)
    sample_files = evenly_spaced_sample(files, args.preflight_sample_files)
    real_sample = validate_real_sample(sample_files, question_meta, args.extreme_elapsed_ms)
    space_estimate = estimate_output_space(
        sample_files=sample_files,
        question_meta=question_meta,
        output_format=args.output_format,
        extreme_elapsed_ms=args.extreme_elapsed_ms,
        total_input_bytes=int(kt1_summary["total_input_bytes"]),
    )
    disk = shutil.disk_usage(output_dir)
    required = int(space_estimate["estimated_output_bytes_with_safety_factor"])
    if not args.skip_disk_check and disk.free < required:
        raise ProcessingError(
            f"Insufficient free space in {output_dir}. Estimated required: {human_bytes(required)}, "
            f"available: {human_bytes(disk.free)}. Use a larger drive; do not bypass unless independently verified."
        )

    return {
        "python_version": sys.version.split()[0],
        "pyarrow_version": pyarrow_version,
        "self_test": self_test,
        "real_sample": real_sample,
        "space_estimate": space_estimate,
        "disk_free_bytes_before_run": disk.free,
        "disk_free_human": human_bytes(disk.free),
        "estimated_output_human": human_bytes(required),
        "status": "PASS",
    }


def run(args: argparse.Namespace) -> int:
    kt1_dir = args.kt1_dir.resolve()
    contents = args.contents.resolve()
    output_dir = args.output_dir.resolve()

    print(f"[1/6] Loading question metadata from: {contents}", flush=True)
    question_meta, question_summary = load_question_metadata(contents)
    print(
        f"      Loaded {len(question_meta):,} questions; "
        f"missing skill tags: {question_summary['missing_skill_rows']:,}",
        flush=True,
    )

    print(f"[2/6] Scanning KT1 directory: {kt1_dir}", flush=True)
    files, kt1_summary = enumerate_kt1_files(kt1_dir)
    print(
        f"      Found {len(files):,} learner CSV files, "
        f"raw size {human_bytes(kt1_summary['total_input_bytes'])}",
        flush=True,
    )

    if args.max_files is not None:
        if args.max_files <= 0:
            raise ProcessingError("--max-files must be positive")
        files = files[: args.max_files]
        kt1_summary = dict(kt1_summary)
        kt1_summary["processing_file_count"] = len(files)
        kt1_summary["limited_by_max_files"] = True
    else:
        kt1_summary = dict(kt1_summary)
        kt1_summary["processing_file_count"] = len(files)
        kt1_summary["limited_by_max_files"] = False

    run_identity = {
        "directory_fingerprint": kt1_summary["directory_fingerprint"],
        "questions_sha256": question_summary["questions_sha256"],
        "processing_file_count": len(files),
        "first_file": files[0].name,
        "last_file": files[-1].name,
        "output_format": args.output_format,
        "files_per_shard": args.files_per_shard,
        "extreme_elapsed_ms": args.extreme_elapsed_ms,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    progress = load_progress(progress_path)
    check_resume_compatibility(progress, run_identity)

    print("[3/6] Running automatic self-test and real-data preflight...", flush=True)
    preflight = perform_preflight(args, files, question_meta, kt1_summary, output_dir)
    atomic_write_json(output_dir / "preflight_report.json", preflight)
    print(
        f"      PASS. Estimated output: {preflight['estimated_output_human']}; "
        f"free space: {preflight['disk_free_human']}",
        flush=True,
    )

    if args.preflight_only:
        print("Preflight-only mode completed successfully. No full KT1 processing was started.")
        return 0

    progress["run_identity"] = run_identity
    progress["last_started_at_utc"] = utc_now_iso()
    atomic_write_json(progress_path, progress)

    print("[4/6] Writing normalized question metadata...", flush=True)
    q_output = write_question_metadata(output_dir, question_meta, args.output_format)
    print(f"      {q_output}", flush=True)

    total_shards = math.ceil(len(files) / args.files_per_shard)
    print(
        f"[5/6] Processing {len(files):,} learner files in {total_shards:,} atomic shards "
        f"({args.files_per_shard} files/shard, {args.workers} worker threads)...",
        flush=True,
    )

    completed_shards: dict[str, Any] = progress.setdefault("completed_shards", {})
    # Verify existing shards exactly once at startup. Re-hashing every prior shard
    # after every new shard would make a resume run quadratically slower.
    invalid_existing: list[str] = []
    for key, stored_stats in list(completed_shards.items()):
        shard_num = int(key)
        if shard_num < 0 or shard_num >= total_shards or not verify_completed_shard(output_dir, stored_stats):
            invalid_existing.append(key)
    for key in invalid_existing:
        completed_shards.pop(key, None)
    if invalid_existing:
        progress["updated_at_utc"] = utc_now_iso()
        atomic_write_json(progress_path, progress)
        print(f"      Removed {len(invalid_existing)} invalid/incompatible stored shard records.", flush=True)

    done_count = len(completed_shards)
    initial_done_count = done_count
    total_start = time.perf_counter()
    for shard_id in range(total_shards):
        key = str(shard_id)
        start = shard_id * args.files_per_shard
        end = min(start + args.files_per_shard, len(files))
        shard_files = files[start:end]

        existing = completed_shards.get(key)
        if existing:
            print(
                f"      [{shard_id + 1}/{total_shards}] SKIP verified shard "
                f"{shard_id:06d} ({shard_files[0].name}..{shard_files[-1].name})",
                flush=True,
            )
            continue

        stats = process_shard(
            shard_id=shard_id,
            files=shard_files,
            question_meta=question_meta,
            output_dir=output_dir,
            output_format=args.output_format,
            extreme_elapsed_ms=args.extreme_elapsed_ms,
            workers=args.workers,
        )
        completed_shards[key] = asdict(stats)
        progress["last_completed_shard"] = shard_id
        progress["updated_at_utc"] = utc_now_iso()
        atomic_write_json(progress_path, progress)

        done_count += 1
        elapsed_total = time.perf_counter() - total_start
        # Throughput is based on shards completed in this process. If the run was
        # resumed with most shards already done, the estimate remains conservative.
        processed_this_run = max(1, done_count - initial_done_count)
        rate = processed_this_run / elapsed_total if elapsed_total > 0 else 0.0
        remaining = (total_shards - done_count) / rate if rate > 0 else float("nan")
        remaining_text = f"{remaining / 3600:.2f} h" if math.isfinite(remaining) else "unknown"
        print(
            f"      [{shard_id + 1}/{total_shards}] DONE {shard_files[0].name}..{shard_files[-1].name}; "
            f"rows={stats.interaction_rows:,}; file_errors={stats.file_errors}; "
            f"shard_time={stats.elapsed_seconds:.1f}s; estimated_remaining={remaining_text}",
            flush=True,
        )

    print("[6/6] Building and verifying final manifest...", flush=True)
    manifest = build_final_manifest(
        output_dir=output_dir,
        progress=progress,
        kt1_summary=kt1_summary,
        question_summary=question_summary,
        args=args,
        preflight=preflight,
    )
    expected_shards = total_shards
    actual_shards = manifest["output"]["shard_count"]
    if actual_shards != expected_shards:
        raise ProcessingError(
            f"Final verification failed: expected {expected_shards} completed shards, got {actual_shards}"
        )
    if manifest["output"]["learner_rows"] != len(files):
        raise ProcessingError(
            f"Final verification failed: expected {len(files)} learner summary rows, "
            f"got {manifest['output']['learner_rows']}"
        )
    atomic_write_json(output_dir / "manifest.json", manifest)
    print(
        f"COMPLETED. Learners={manifest['output']['learner_rows']:,}; "
        f"interactions={manifest['output']['interaction_rows']:,}; "
        f"file_errors={manifest['output']['file_errors']:,}; manifest={output_dir / 'manifest.json'}",
        flush=True,
    )
    return 0 if manifest["output"]["file_errors"] == 0 else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and convert an extracted EdNet-KT1 directory into resumable sharded outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--kt1-dir", type=Path, required=True, help="Extracted KT1 directory containing u*.csv")
    parser.add_argument(
        "--contents",
        type=Path,
        required=True,
        help="EdNet-Contents.zip, extracted contents directory, or questions.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="New or resumable output directory")
    parser.add_argument(
        "--output-format",
        choices=("parquet", "csv-gzip"),
        default="parquet",
        help="Parquet is recommended for the research pipeline",
    )
    parser.add_argument("--files-per-shard", type=int, default=1000, help="Atomic resume unit")
    parser.add_argument("--workers", type=int, default=min(8, max(1, os.cpu_count() or 1)), help="I/O worker threads")
    parser.add_argument(
        "--extreme-elapsed-ms",
        type=int,
        default=3_600_000,
        help="Elapsed times above this value are retained but flagged",
    )
    parser.add_argument("--preflight-sample-files", type=int, default=30, help="Evenly spaced real files used by preflight")
    parser.add_argument("--preflight-only", action="store_true", help="Run all tests and checks but do not process full KT1")
    parser.add_argument("--skip-disk-check", action="store_true", help="Bypass free-space refusal; not recommended")
    parser.add_argument("--max-files", type=int, default=None, help="Testing only: process the first N files")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.files_per_shard <= 0:
        parser.error("--files-per-shard must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.extreme_elapsed_ms <= 0:
        parser.error("--extreme-elapsed-ms must be positive")
    if args.preflight_sample_files < 3:
        parser.error("--preflight-sample-files must be at least 3")
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Completed shards remain valid; rerun the same command to resume.", file=sys.stderr)
        return 130
    except ProcessingError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"UNEXPECTED FATAL ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
