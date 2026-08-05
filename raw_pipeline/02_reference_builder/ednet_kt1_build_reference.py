#!/usr/bin/env python3
"""Build a reproducible 20-skill EdNet-KT1 BKT-F reference dataset.

Input: the completed output directory of ednet_kt1_prepare.py.
Output: exact selection diagnostics, deterministic learner splits, a compact
calibration/evaluation interaction sample, manifests, and one uploadable ZIP.

The program is resumable at shard level. It never modifies the source data.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# Stable EdNet quality bits from ednet_kt1_prepare.py.
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

# QUESTION_BEFORE_DEPLOYMENT is deliberately NOT excluded. EdNet timestamps
# were privacy-shifted and are not proven comparable with deployed_at.
REFERENCE_EXCLUDE_MASK = (
    Q_UNKNOWN_QUESTION
    | Q_MISSING_CORRECT_ANSWER
    | Q_MISSING_SKILL_TAG
    | Q_DUPLICATE_EVENT
    | Q_INVALID_RESPONSE
    | Q_INVALID_TIMESTAMP
)

SPLIT_SALT = np.uint64(0xA17E5D93C4B28601)
SAMPLE_SALT = np.uint64(0x6C8E9CF570932BD5)
GAP_SALT = np.uint64(0xD1B54A32D192ED03)
UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)

SPLIT_NAMES = ("train", "calibration", "test")
SPLIT_CODE_TO_NAME = {0: "train", 1: "calibration", 2: "test"}
SAMPLE_CLASS_NAMES = {0: "cold_1_2", 1: "repeat_3_plus"}

# Positive-gap bins. Left-closed, right-open, except the last open-ended bin.
GAP_BIN_EDGES_MS = np.array([
    0,
    60_000,
    5 * 60_000,
    30 * 60_000,
    2 * 60 * 60_000,
    12 * 60 * 60_000,
    24 * 60 * 60_000,
    3 * 24 * 60 * 60_000,
    7 * 24 * 60 * 60_000,
    30 * 24 * 60 * 60_000,
], dtype=np.int64)
GAP_BIN_LABELS = (
    "gt0_lt1min",
    "1min_lt5min",
    "5min_lt30min",
    "30min_lt2h",
    "2h_lt12h",
    "12h_lt1d",
    "1d_lt3d",
    "3d_lt7d",
    "7d_lt30d",
    "30d_plus",
)
EVENT_COUNT_BINS = (
    ("1", 1, 1),
    ("2", 2, 2),
    ("3_4", 3, 4),
    ("5_9", 5, 9),
    ("10_19", 10, 19),
    ("20_plus", 20, None),
)

INTERACTION_REQUIRED_COLUMNS = (
    "learner_id",
    "sequence_index",
    "timestamp_ms",
    "question_id",
    "is_correct",
    "elapsed_time_ms",
    "part",
    "skill_count",
    "primary_skill_id",
    "quality_flags",
)
LEARNER_REQUIRED_COLUMNS = ("learner_id",)
QUESTION_REQUIRED_COLUMNS = (
    "question_id",
    "part",
    "skill_count",
    "primary_skill_id",
)


class ReferenceBuildError(RuntimeError):
    """Fatal configuration, integrity, or processing error."""


@dataclass(frozen=True)
class BuildConfig:
    reference_skill_count: int = 20
    min_item_interactions: int = 200
    min_items_per_skill: int = 50
    min_unique_learners: int = 10_000
    min_repeat_learners: int = 2_000
    min_positive_gaps: int = 10_000
    min_item_accuracy_spread: float = 0.20
    min_skill_accuracy: float = 0.20
    max_skill_accuracy: float = 0.95
    relaxed_min_unique_learners: int = 5_000
    relaxed_min_repeat_learners: int = 1_000
    relaxed_min_positive_gaps: int = 5_000
    relaxed_min_item_accuracy_spread: float = 0.15
    repeat_train_quota: int = 5_000
    repeat_calibration_quota: int = 2_000
    repeat_test_quota: int = 2_000
    cold_quota_per_split: int = 500
    gap_sample_per_shard_skill: int = 200
    gap_sample_final_per_skill: int = 100_000
    compression: str = "zstd"
    compression_level: int = 6


@dataclass(frozen=True)
class SourceShard:
    shard_id: int
    interaction_path: Path
    learner_path: Path
    interaction_rows: int
    learner_rows: int
    interaction_sha256: str
    learner_sha256: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def require_pyarrow() -> tuple[Any, Any, Any]:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.compute as pc  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        raise ReferenceBuildError(
            "PyArrow is required. Use the supplied PowerShell launcher and the "
            "existing EdNet processing environment (pyarrow==25.0.0)."
        ) from exc
    return pa, pc, pq


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding=encoding, newline="") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_relpath(value: str) -> Path:
    return Path(value.replace("\\", "/"))


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    number = float(value)
    for unit in units:
        if number < 1024.0 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024.0
    return f"{number:.2f} TiB"


def splitmix64_array(values: np.ndarray) -> np.ndarray:
    z = values.astype(np.uint64, copy=True)
    z = (z + np.uint64(0x9E3779B97F4A7C15)) & UINT64_MASK
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & UINT64_MASK
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & UINT64_MASK
    return (z ^ (z >> np.uint64(31))) & UINT64_MASK


def learner_split_codes(learner_ids: np.ndarray) -> np.ndarray:
    hashes = splitmix64_array(learner_ids.astype(np.uint64) ^ SPLIT_SALT)
    buckets = (hashes % np.uint64(10_000)).astype(np.int16)
    return np.where(buckets < 7_000, 0, np.where(buckets < 8_500, 1, 2)).astype(np.int8)


def sample_priorities(learner_ids: np.ndarray, skill_ids: np.ndarray, class_codes: np.ndarray) -> np.ndarray:
    values = (
        learner_ids.astype(np.uint64)
        ^ (skill_ids.astype(np.uint64) << np.uint64(32))
        ^ (class_codes.astype(np.uint64) << np.uint64(60))
        ^ SAMPLE_SALT
    )
    return splitmix64_array(values)


def gap_priorities(learner_ids: np.ndarray, skill_ids: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    values = (
        learner_ids.astype(np.uint64)
        ^ (skill_ids.astype(np.uint64) << np.uint64(32))
        ^ timestamps.astype(np.uint64)
        ^ GAP_SALT
    )
    return splitmix64_array(values)


def event_count_bin_codes(counts: np.ndarray) -> np.ndarray:
    output = np.full(counts.shape, -1, dtype=np.int8)
    for index, (_, lower, upper) in enumerate(EVENT_COUNT_BINS):
        mask = counts >= lower
        if upper is not None:
            mask &= counts <= upper
        output[mask] = index
    if np.any(output < 0):
        raise ReferenceBuildError("Internal error: unclassified learner-skill event count")
    return output


def top_k_indices(priorities: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or priorities.size == 0:
        return np.empty(0, dtype=np.int64)
    if priorities.size <= k:
        return np.argsort(priorities, kind="stable")
    selected = np.argpartition(priorities, k - 1)[:k]
    return selected[np.argsort(priorities[selected], kind="stable")]


def minmax(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    output = np.zeros(values.shape, dtype=np.float64)
    if not np.any(finite):
        return output
    lo = float(np.min(values[finite]))
    hi = float(np.max(values[finite]))
    if hi <= lo:
        output[finite] = 1.0
    else:
        output[finite] = (values[finite] - lo) / (hi - lo)
    return output


def selection_scores(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if not rows:
        return np.empty(0, dtype=np.float64)
    learners = np.log1p(np.array([float(r["unique_learners"]) for r in rows]))
    repeat = np.log1p(np.array([float(r["learners_3_plus"]) for r in rows]))
    gaps = np.log1p(np.array([float(r["positive_gap_count"]) for r in rows]))
    items = np.log1p(np.array([float(r["eligible_items"]) for r in rows]))
    spread = np.array([float(r["item_accuracy_p90_p10_spread"]) for r in rows])
    interactions = np.log1p(np.array([float(r["interaction_rows"]) for r in rows]))
    return (
        0.25 * minmax(learners)
        + 0.25 * minmax(repeat)
        + 0.20 * minmax(gaps)
        + 0.15 * minmax(items)
        + 0.10 * minmax(spread)
        + 0.05 * minmax(interactions)
    )


def safe_numpy(column: Any, fill_value: Any, dtype: Any) -> np.ndarray:
    _, pc, _ = require_pyarrow()
    array = column.combine_chunks() if hasattr(column, "combine_chunks") else column
    filled = pc.fill_null(array, fill_value)
    return np.asarray(filled.to_numpy(zero_copy_only=False), dtype=dtype)


def valid_numpy(column: Any) -> np.ndarray:
    _, pc, _ = require_pyarrow()
    array = column.combine_chunks() if hasattr(column, "combine_chunks") else column
    return np.asarray(pc.is_valid(array).to_numpy(zero_copy_only=False), dtype=np.bool_)


def question_numbers_filtered(question_column: Any, mask: np.ndarray) -> np.ndarray:
    pa, pc, _ = require_pyarrow()
    array = question_column.combine_chunks() if hasattr(question_column, "combine_chunks") else question_column
    filtered = pc.filter(array, pa.array(mask))
    suffix = pc.utf8_slice_codeunits(filtered, start=1)
    try:
        numeric = pc.cast(suffix, pa.int32(), safe=True)
    except Exception as exc:
        raise ReferenceBuildError("A retained question_id is not in q<integer> form") from exc
    return np.asarray(numeric.to_numpy(zero_copy_only=False), dtype=np.int32)


def parquet_file_info(path: Path) -> tuple[int, Any]:
    """Return row count and Arrow schema, closing the native file handle explicitly.

    Explicit close is required on Windows because an open ParquetFile can keep the
    file locked and make os.replace(), unlink(), or TemporaryDirectory cleanup fail.
    """
    _, _, pq = require_pyarrow()
    parquet_file = pq.ParquetFile(path)
    try:
        metadata = parquet_file.metadata
        if metadata is None:
            raise ReferenceBuildError(f"Parquet metadata are unavailable: {path}")
        row_count = int(metadata.num_rows)
        schema = parquet_file.schema_arrow
        return row_count, schema
    finally:
        parquet_file.close(force=True)


def write_parquet_atomic(table: Any, path: Path, compression: str, compression_level: int) -> None:
    _, _, pq = require_pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    try:
        pq.write_table(
            table,
            tmp,
            compression=compression,
            compression_level=compression_level,
            use_dictionary=True,
            write_statistics=True,
        )
        row_count, _ = parquet_file_info(tmp)
        if row_count != table.num_rows:
            raise ReferenceBuildError(f"Parquet verification failed: {path}")
        os.replace(tmp, path)
    except Exception:
        gc.collect()
        try:
            tmp.unlink(missing_ok=True)
        except PermissionError:
            pass
        raise


def write_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_source(input_dir: Path) -> tuple[dict[str, Any], list[SourceShard]]:
    _, _, pq = require_pyarrow()
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        raise ReferenceBuildError(f"Missing source manifest: {manifest_path}")
    manifest = load_json(manifest_path)
    shard_records = manifest.get("shards")
    if not isinstance(shard_records, list) or not shard_records:
        raise ReferenceBuildError("Source manifest contains no shards")
    shards: list[SourceShard] = []
    expected_id = 0
    for row in shard_records:
        shard_id = int(row["shard_id"])
        if shard_id != expected_id:
            raise ReferenceBuildError(f"Non-contiguous source shard sequence at {shard_id}, expected {expected_id}")
        expected_id += 1
        interaction_path = input_dir / canonical_relpath(str(row["interaction_output"]))
        learner_path = input_dir / canonical_relpath(str(row["learner_output"]))
        if not interaction_path.exists():
            raise ReferenceBuildError(f"Missing interaction shard: {interaction_path}")
        if not learner_path.exists():
            raise ReferenceBuildError(f"Missing learner shard: {learner_path}")
        i_rows, i_schema = parquet_file_info(interaction_path)
        l_rows, l_schema = parquet_file_info(learner_path)
        if i_rows != int(row["interaction_rows"]):
            raise ReferenceBuildError(f"Interaction row-count mismatch: {interaction_path}")
        if l_rows != int(row["learner_rows"]):
            raise ReferenceBuildError(f"Learner row-count mismatch: {learner_path}")
        i_names = i_schema.names
        l_names = l_schema.names
        missing_i = [c for c in INTERACTION_REQUIRED_COLUMNS if c not in i_names]
        missing_l = [c for c in LEARNER_REQUIRED_COLUMNS if c not in l_names]
        if missing_i:
            raise ReferenceBuildError(f"Interaction shard missing columns {missing_i}: {interaction_path}")
        if missing_l:
            raise ReferenceBuildError(f"Learner shard missing columns {missing_l}: {learner_path}")
        shards.append(SourceShard(
            shard_id=shard_id,
            interaction_path=interaction_path,
            learner_path=learner_path,
            interaction_rows=int(row["interaction_rows"]),
            learner_rows=int(row["learner_rows"]),
            interaction_sha256=str(row["interaction_sha256"]),
            learner_sha256=str(row["learner_sha256"]),
        ))
    return manifest, shards


def load_question_metadata(input_dir: Path) -> dict[str, Any]:
    _, _, pq = require_pyarrow()
    path = input_dir / "metadata" / "questions.parquet"
    if not path.exists():
        raise ReferenceBuildError(f"Missing questions metadata: {path}")
    table = pq.read_table(path)
    missing = [c for c in QUESTION_REQUIRED_COLUMNS if c not in table.column_names]
    if missing:
        raise ReferenceBuildError(f"questions.parquet missing columns: {missing}")
    qids = table.column("question_id").to_pylist()
    qnums: list[int] = []
    for qid in qids:
        text = str(qid)
        if not text.startswith("q") or not text[1:].isdigit():
            raise ReferenceBuildError(f"Invalid question_id in metadata: {qid!r}")
        qnums.append(int(text[1:]))
    qmax = max(qnums)
    skill = safe_numpy(table.column("primary_skill_id"), -1, np.int32)
    skill_count = safe_numpy(table.column("skill_count"), 0, np.int16)
    part = safe_numpy(table.column("part"), -1, np.int16)
    q_to_skill = np.full(qmax + 1, -1, dtype=np.int32)
    q_to_skill_count = np.zeros(qmax + 1, dtype=np.int16)
    q_to_part = np.full(qmax + 1, -1, dtype=np.int16)
    q_exists = np.zeros(qmax + 1, dtype=np.bool_)
    for idx, qnum in enumerate(qnums):
        if q_exists[qnum]:
            raise ReferenceBuildError(f"Duplicate numeric question ID: q{qnum}")
        q_exists[qnum] = True
        q_to_skill[qnum] = int(skill[idx])
        q_to_skill_count[qnum] = int(skill_count[idx])
        q_to_part[qnum] = int(part[idx])
    return {
        "path": path,
        "table": table,
        "qmax": qmax,
        "q_exists": q_exists,
        "q_to_skill": q_to_skill,
        "q_to_skill_count": q_to_skill_count,
        "q_to_part": q_to_part,
        "max_skill_id": int(max(0, np.max(skill))),
    }


def base_reference_mask(table: Any) -> np.ndarray:
    skill_count = safe_numpy(table.column("skill_count"), 0, np.int16)
    primary_skill = safe_numpy(table.column("primary_skill_id"), -1, np.int32)
    flags = safe_numpy(table.column("quality_flags"), 0, np.uint16)
    correctness_valid = valid_numpy(table.column("is_correct"))
    timestamp_valid = valid_numpy(table.column("timestamp_ms"))
    return (
        (skill_count == 1)
        & (primary_skill >= 0)
        & correctness_valid
        & timestamp_valid
        & ((flags & np.uint16(REFERENCE_EXCLUDE_MASK)) == 0)
    )


def pass1_partial_path(work_dir: Path, shard_id: int) -> Path:
    return work_dir / "pass1_item_stats" / f"part-{shard_id:06d}.npz"


def pass2_group_path(work_dir: Path, shard_id: int) -> Path:
    return work_dir / "pass2_groups" / f"part-{shard_id:06d}.parquet"


def pass2_gap_path(work_dir: Path, shard_id: int) -> Path:
    return work_dir / "pass2_gaps" / f"part-{shard_id:06d}.npz"


def pass3_path(work_dir: Path, shard_id: int) -> Path:
    return work_dir / "pass3_reference" / f"part-{shard_id:06d}.parquet"




def pass1_checkpoint_valid(path: Path, shard: SourceShard, qmax: int) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                str(data["source_interaction_sha256"].item()) == shard.interaction_sha256
                and data["item_rows"].shape == (qmax + 1,)
                and data["item_correct"].shape == (qmax + 1,)
            )
    except Exception:
        return False


def pass2_checkpoint_valid(
    group_path: Path, gap_path: Path, shard: SourceShard, candidate_skills: Sequence[int]
) -> bool:
    try:
        _, _, pq = require_pyarrow()
        required = {
            "learner_id", "skill_id", "split_code", "event_count", "correct_count",
            "first_timestamp_ms", "last_timestamp_ms", "positive_gap_count",
            "max_positive_gap_ms",
        }
        if not group_path.exists() or not gap_path.exists():
            return False
        _, group_schema = parquet_file_info(group_path)
        if not required.issubset(set(group_schema.names)):
            return False
        with np.load(gap_path, allow_pickle=False) as data:
            if str(data["source_interaction_sha256"].item()) != shard.interaction_sha256:
                return False
            if "candidate_skills" in data and not np.array_equal(
                data["candidate_skills"], np.array(candidate_skills, dtype=np.int32)
            ):
                return False
            if "group_sha256" not in data:
                return False
            if str(data["group_sha256"].item()) != sha256_file(group_path):
                return False
        return True
    except Exception:
        return False


def pass3_checkpoint_valid(path: Path) -> bool:
    try:
        if not path.exists():
            return False
        _, schema = parquet_file_info(path)
        return schema == reference_output_schema()
    except Exception:
        return False


def process_pass1_shard(shard: SourceShard, qmax: int, output_path: Path) -> None:
    _, _, pq = require_pyarrow()
    table = pq.read_table(
        shard.interaction_path,
        columns=["question_id", "primary_skill_id", "skill_count", "is_correct", "timestamp_ms", "quality_flags"],
    )
    mask = base_reference_mask(table)
    qnums = question_numbers_filtered(table.column("question_id"), mask)
    correct_all = safe_numpy(table.column("is_correct"), False, np.bool_)
    correct = correct_all[mask].astype(np.int64)
    if np.any(qnums < 0) or np.any(qnums > qmax):
        raise ReferenceBuildError(f"Question numeric ID outside metadata range in shard {shard.shard_id}")
    item_rows = np.bincount(qnums, minlength=qmax + 1).astype(np.int64)
    item_correct = np.bincount(qnums, weights=correct, minlength=qmax + 1).astype(np.int64)
    write_npz_atomic(
        output_path,
        source_interaction_sha256=np.array(shard.interaction_sha256),
        source_rows=np.array(shard.interaction_rows, dtype=np.int64),
        retained_rows=np.array(qnums.size, dtype=np.int64),
        item_rows=item_rows,
        item_correct=item_correct,
    )


def merge_pass1(shards: Sequence[SourceShard], work_dir: Path, qmax: int) -> tuple[np.ndarray, np.ndarray, int]:
    rows = np.zeros(qmax + 1, dtype=np.int64)
    correct = np.zeros(qmax + 1, dtype=np.int64)
    retained = 0
    for shard in shards:
        path = pass1_partial_path(work_dir, shard.shard_id)
        if not path.exists():
            raise ReferenceBuildError(f"Missing pass-1 checkpoint: {path}")
        with np.load(path, allow_pickle=False) as data:
            if str(data["source_interaction_sha256"].item()) != shard.interaction_sha256:
                raise ReferenceBuildError(f"Stale pass-1 checkpoint: {path}")
            part_rows = data["item_rows"]
            part_correct = data["item_correct"]
            if part_rows.shape != rows.shape or part_correct.shape != correct.shape:
                raise ReferenceBuildError(f"Invalid pass-1 array shape: {path}")
            rows += part_rows
            correct += part_correct
            retained += int(data["retained_rows"].item())
    if int(rows.sum()) != retained:
        raise ReferenceBuildError("Pass-1 merge count mismatch")
    return rows, correct, retained


def candidate_items(
    item_rows: np.ndarray,
    item_correct: np.ndarray,
    metadata: Mapping[str, Any],
    config: BuildConfig,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    q_to_skill = metadata["q_to_skill"]
    q_to_skill_count = metadata["q_to_skill_count"]
    eligible = (
        (item_rows >= config.min_item_interactions)
        & (q_to_skill_count == 1)
        & (q_to_skill >= 0)
    )
    max_skill = metadata["max_skill_id"]
    item_count_by_skill = np.bincount(
        q_to_skill[eligible], minlength=max_skill + 1
    ).astype(np.int64)
    candidate_skills = np.flatnonzero(item_count_by_skill >= config.min_items_per_skill).astype(int).tolist()
    if len(candidate_skills) < config.reference_skill_count:
        raise ReferenceBuildError(
            f"Only {len(candidate_skills)} skills have at least {config.min_items_per_skill} "
            f"items with at least {config.min_item_interactions} interactions; "
            f"{config.reference_skill_count} are required."
        )
    candidate_item_mask = eligible & np.isin(q_to_skill, np.array(candidate_skills, dtype=np.int32))
    return eligible, candidate_item_mask, candidate_skills


def filter_candidate_arrays(table: Any, candidate_item_mask: np.ndarray) -> dict[str, np.ndarray]:
    base = base_reference_mask(table)
    qnums_base = question_numbers_filtered(table.column("question_id"), base)
    in_range = (qnums_base >= 0) & (qnums_base < candidate_item_mask.size)
    retained_base_indices = np.flatnonzero(base)
    eligible_local = np.zeros(qnums_base.shape, dtype=np.bool_)
    eligible_local[in_range] = candidate_item_mask[qnums_base[in_range]]
    final_indices = retained_base_indices[eligible_local]
    return {
        "learner_id": safe_numpy(table.column("learner_id"), -1, np.int32)[final_indices],
        "sequence_index": safe_numpy(table.column("sequence_index"), -1, np.int32)[final_indices],
        "timestamp_ms": safe_numpy(table.column("timestamp_ms"), -1, np.int64)[final_indices],
        "question_num": qnums_base[eligible_local],
        "skill_id": safe_numpy(table.column("primary_skill_id"), -1, np.int32)[final_indices],
        "is_correct": safe_numpy(table.column("is_correct"), False, np.bool_)[final_indices],
        "source_indices": final_indices.astype(np.int64),
    }


def process_pass2_shard(
    shard: SourceShard,
    candidate_item_mask: np.ndarray,
    candidate_skills: Sequence[int],
    group_output: Path,
    gap_output: Path,
    config: BuildConfig,
) -> None:
    pa, _, pq = require_pyarrow()
    table = pq.read_table(
        shard.interaction_path,
        columns=[
            "learner_id", "sequence_index", "timestamp_ms", "question_id",
            "primary_skill_id", "skill_count", "is_correct", "quality_flags",
        ],
    )
    arrays = filter_candidate_arrays(table, candidate_item_mask)
    learner = arrays["learner_id"]
    skill = arrays["skill_id"]
    timestamp = arrays["timestamp_ms"]
    sequence = arrays["sequence_index"]
    correct = arrays["is_correct"].astype(np.int64)
    candidate_skills_array = np.array(candidate_skills, dtype=np.int32)
    skill_to_index = {int(s): i for i, s in enumerate(candidate_skills)}
    gap_hist = np.zeros((len(candidate_skills), len(GAP_BIN_LABELS)), dtype=np.int64)
    sampled_skill: list[np.ndarray] = []
    sampled_gap: list[np.ndarray] = []
    sampled_priority: list[np.ndarray] = []

    if learner.size == 0:
        empty = pa.table({
            "learner_id": pa.array([], type=pa.int32()),
            "skill_id": pa.array([], type=pa.int32()),
            "split_code": pa.array([], type=pa.int8()),
            "event_count": pa.array([], type=pa.int32()),
            "correct_count": pa.array([], type=pa.int32()),
            "first_timestamp_ms": pa.array([], type=pa.int64()),
            "last_timestamp_ms": pa.array([], type=pa.int64()),
            "positive_gap_count": pa.array([], type=pa.int32()),
            "max_positive_gap_ms": pa.array([], type=pa.int64()),
        })
        write_parquet_atomic(empty, group_output, config.compression, config.compression_level)
        write_npz_atomic(
            gap_output,
            source_interaction_sha256=np.array(shard.interaction_sha256),
            gap_hist=gap_hist,
            sampled_skill_id=np.empty(0, dtype=np.int32),
            sampled_gap_ms=np.empty(0, dtype=np.int64),
            sampled_priority=np.empty(0, dtype=np.uint64),
            group_sha256=np.array(sha256_file(group_output)),
        )
        return

    order = np.lexsort((sequence, timestamp, skill, learner))
    learner = learner[order]
    skill = skill[order]
    timestamp = timestamp[order]
    correct = correct[order]
    group_start_mask = np.ones(learner.size, dtype=np.bool_)
    group_start_mask[1:] = (learner[1:] != learner[:-1]) | (skill[1:] != skill[:-1])
    starts = np.flatnonzero(group_start_mask)
    ends = np.r_[starts[1:], learner.size]
    event_count = (ends - starts).astype(np.int32)
    correct_count = np.add.reduceat(correct, starts).astype(np.int32)
    group_learner = learner[starts]
    group_skill = skill[starts]
    first_ts = timestamp[starts]
    last_ts = timestamp[ends - 1]

    gap_at_row = np.zeros(learner.size, dtype=np.int64)
    same_pair = (learner[1:] == learner[:-1]) & (skill[1:] == skill[:-1])
    raw_gaps = timestamp[1:] - timestamp[:-1]
    positive = same_pair & (raw_gaps > 0)
    positive_positions = np.flatnonzero(positive) + 1
    gap_at_row[positive_positions] = raw_gaps[positive]
    positive_indicator = (gap_at_row > 0).astype(np.int32)
    positive_gap_count = np.add.reduceat(positive_indicator, starts).astype(np.int32)
    max_positive_gap = np.maximum.reduceat(gap_at_row, starts).astype(np.int64)

    if np.any(positive):
        gap_values = raw_gaps[positive].astype(np.int64)
        gap_skill = skill[1:][positive].astype(np.int32)
        gap_learner = learner[1:][positive].astype(np.int32)
        gap_timestamp = timestamp[1:][positive].astype(np.int64)
        bin_codes = np.searchsorted(GAP_BIN_EDGES_MS[1:], gap_values, side="right")
        for skill_value in np.unique(gap_skill):
            local = gap_skill == skill_value
            skill_index = skill_to_index[int(skill_value)]
            counts = np.bincount(bin_codes[local], minlength=len(GAP_BIN_LABELS))
            gap_hist[skill_index] += counts[: len(GAP_BIN_LABELS)]
            priorities = gap_priorities(gap_learner[local], gap_skill[local], gap_timestamp[local])
            chosen = top_k_indices(priorities, config.gap_sample_per_shard_skill)
            sampled_skill.append(gap_skill[local][chosen])
            sampled_gap.append(gap_values[local][chosen])
            sampled_priority.append(priorities[chosen])

    group_table = pa.table({
        "learner_id": pa.array(group_learner, type=pa.int32()),
        "skill_id": pa.array(group_skill, type=pa.int32()),
        "split_code": pa.array(learner_split_codes(group_learner), type=pa.int8()),
        "event_count": pa.array(event_count, type=pa.int32()),
        "correct_count": pa.array(correct_count, type=pa.int32()),
        "first_timestamp_ms": pa.array(first_ts, type=pa.int64()),
        "last_timestamp_ms": pa.array(last_ts, type=pa.int64()),
        "positive_gap_count": pa.array(positive_gap_count, type=pa.int32()),
        "max_positive_gap_ms": pa.array(max_positive_gap, type=pa.int64()),
    })
    write_parquet_atomic(group_table, group_output, config.compression, config.compression_level)
    write_npz_atomic(
        gap_output,
        source_interaction_sha256=np.array(shard.interaction_sha256),
        candidate_skills=candidate_skills_array,
        gap_hist=gap_hist,
        sampled_skill_id=np.concatenate(sampled_skill) if sampled_skill else np.empty(0, dtype=np.int32),
        sampled_gap_ms=np.concatenate(sampled_gap) if sampled_gap else np.empty(0, dtype=np.int64),
        sampled_priority=np.concatenate(sampled_priority) if sampled_priority else np.empty(0, dtype=np.uint64),
        group_sha256=np.array(sha256_file(group_output)),
    )


def aggregate_pass2(
    shards: Sequence[SourceShard],
    work_dir: Path,
    candidate_skills: Sequence[int],
) -> tuple[dict[int, dict[str, Any]], np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]]]:
    _, _, pq = require_pyarrow()
    index = {int(skill): i for i, skill in enumerate(candidate_skills)}
    n = len(candidate_skills)
    unique_learners = np.zeros(n, dtype=np.int64)
    interactions = np.zeros(n, dtype=np.int64)
    correct = np.zeros(n, dtype=np.int64)
    learners_2_plus = np.zeros(n, dtype=np.int64)
    learners_3_plus = np.zeros(n, dtype=np.int64)
    learners_5_plus = np.zeros(n, dtype=np.int64)
    positive_gaps = np.zeros(n, dtype=np.int64)
    split_learners = np.zeros((n, 3), dtype=np.int64)
    split_interactions = np.zeros((n, 3), dtype=np.int64)
    event_bins = np.zeros((n, len(EVENT_COUNT_BINS)), dtype=np.int64)
    gap_hist = np.zeros((n, len(GAP_BIN_LABELS)), dtype=np.int64)
    gap_values_by_skill: dict[int, list[np.ndarray]] = {int(s): [] for s in candidate_skills}
    gap_priorities_by_skill: dict[int, list[np.ndarray]] = {int(s): [] for s in candidate_skills}

    for shard in shards:
        group_path = pass2_group_path(work_dir, shard.shard_id)
        gap_path = pass2_gap_path(work_dir, shard.shard_id)
        if not group_path.exists() or not gap_path.exists():
            raise ReferenceBuildError(f"Missing pass-2 checkpoint for shard {shard.shard_id}")
        table = pq.read_table(group_path)
        skill = safe_numpy(table.column("skill_id"), -1, np.int32)
        counts = safe_numpy(table.column("event_count"), 0, np.int32)
        corr = safe_numpy(table.column("correct_count"), 0, np.int32)
        split = safe_numpy(table.column("split_code"), -1, np.int8)
        gaps = safe_numpy(table.column("positive_gap_count"), 0, np.int32)
        bin_codes = event_count_bin_codes(counts) if counts.size else np.empty(0, dtype=np.int8)
        for skill_value in np.unique(skill):
            if int(skill_value) not in index:
                raise ReferenceBuildError(f"Unexpected skill in pass-2 groups: {skill_value}")
            i = index[int(skill_value)]
            local = skill == skill_value
            local_counts = counts[local]
            local_split = split[local]
            unique_learners[i] += int(local.sum())
            interactions[i] += int(local_counts.sum())
            correct[i] += int(corr[local].sum())
            learners_2_plus[i] += int(np.sum(local_counts >= 2))
            learners_3_plus[i] += int(np.sum(local_counts >= 3))
            learners_5_plus[i] += int(np.sum(local_counts >= 5))
            positive_gaps[i] += int(gaps[local].sum())
            for split_code in range(3):
                sm = local_split == split_code
                split_learners[i, split_code] += int(sm.sum())
                split_interactions[i, split_code] += int(local_counts[sm].sum())
            local_bins = bin_codes[local]
            event_bins[i] += np.bincount(local_bins, minlength=len(EVENT_COUNT_BINS))[: len(EVENT_COUNT_BINS)]
        with np.load(gap_path, allow_pickle=False) as data:
            if str(data["source_interaction_sha256"].item()) != shard.interaction_sha256:
                raise ReferenceBuildError(f"Stale pass-2 gap checkpoint: {gap_path}")
            if "candidate_skills" in data and not np.array_equal(data["candidate_skills"], np.array(candidate_skills)):
                raise ReferenceBuildError(f"Candidate-skill mismatch: {gap_path}")
            gap_hist += data["gap_hist"]
            sampled_skill = data["sampled_skill_id"]
            sampled_gap = data["sampled_gap_ms"]
            sampled_priority = data["sampled_priority"]
            for skill_value in np.unique(sampled_skill):
                local = sampled_skill == skill_value
                gap_values_by_skill[int(skill_value)].append(sampled_gap[local])
                gap_priorities_by_skill[int(skill_value)].append(sampled_priority[local])

    stats: dict[int, dict[str, Any]] = {}
    gap_samples: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for skill_value in candidate_skills:
        i = index[int(skill_value)]
        stats[int(skill_value)] = {
            "skill_id": int(skill_value),
            "unique_learners": int(unique_learners[i]),
            "interaction_rows": int(interactions[i]),
            "correct_rows": int(correct[i]),
            "accuracy": float(correct[i] / interactions[i]) if interactions[i] else math.nan,
            "learners_2_plus": int(learners_2_plus[i]),
            "learners_3_plus": int(learners_3_plus[i]),
            "learners_5_plus": int(learners_5_plus[i]),
            "positive_gap_count": int(positive_gaps[i]),
            "train_learners": int(split_learners[i, 0]),
            "calibration_learners": int(split_learners[i, 1]),
            "test_learners": int(split_learners[i, 2]),
            "train_interactions": int(split_interactions[i, 0]),
            "calibration_interactions": int(split_interactions[i, 1]),
            "test_interactions": int(split_interactions[i, 2]),
        }
        for b, (label, _, _) in enumerate(EVENT_COUNT_BINS):
            stats[int(skill_value)][f"learners_events_{label}"] = int(event_bins[i, b])
        values = np.concatenate(gap_values_by_skill[int(skill_value)]) if gap_values_by_skill[int(skill_value)] else np.empty(0, dtype=np.int64)
        priorities = np.concatenate(gap_priorities_by_skill[int(skill_value)]) if gap_priorities_by_skill[int(skill_value)] else np.empty(0, dtype=np.uint64)
        gap_samples[int(skill_value)] = (values, priorities)
    return stats, gap_hist, gap_samples


def add_item_distribution_stats(
    stats: dict[int, dict[str, Any]],
    eligible_item_mask: np.ndarray,
    candidate_item_mask: np.ndarray,
    item_rows: np.ndarray,
    item_correct: np.ndarray,
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    q_to_skill = metadata["q_to_skill"]
    q_to_part = metadata["q_to_part"]
    output_items: list[dict[str, Any]] = []
    qnums = np.flatnonzero(candidate_item_mask)
    for qnum in qnums:
        skill = int(q_to_skill[qnum])
        rows = int(item_rows[qnum])
        correct = int(item_correct[qnum])
        accuracy = correct / rows if rows else math.nan
        output_items.append({
            "question_id": f"q{qnum}",
            "question_num": int(qnum),
            "skill_id": skill,
            "part": int(q_to_part[qnum]),
            "interaction_rows": rows,
            "correct_rows": correct,
            "accuracy": accuracy,
            "eligible_item": bool(eligible_item_mask[qnum]),
        })
    for skill, row in stats.items():
        local = [x for x in output_items if x["skill_id"] == skill and x["eligible_item"]]
        accuracies = np.array([x["accuracy"] for x in local], dtype=np.float64)
        parts = sorted({int(x["part"]) for x in local if int(x["part"]) >= 0})
        row["eligible_items"] = len(local)
        row["item_accuracy_p10"] = float(np.quantile(accuracies, 0.10)) if accuracies.size else math.nan
        row["item_accuracy_p50"] = float(np.quantile(accuracies, 0.50)) if accuracies.size else math.nan
        row["item_accuracy_p90"] = float(np.quantile(accuracies, 0.90)) if accuracies.size else math.nan
        row["item_accuracy_p90_p10_spread"] = (
            row["item_accuracy_p90"] - row["item_accuracy_p10"] if accuracies.size else math.nan
        )
        row["parts"] = ";".join(str(p) for p in parts)
        row["part_count"] = len(parts)
    return output_items


def choose_reference_skills(stats: dict[int, dict[str, Any]], config: BuildConfig) -> tuple[list[int], str]:
    rows = [stats[key] for key in sorted(stats)]
    strict: list[dict[str, Any]] = []
    relaxed: list[dict[str, Any]] = []
    for row in rows:
        strict_ok = (
            row["eligible_items"] >= config.min_items_per_skill
            and row["unique_learners"] >= config.min_unique_learners
            and row["learners_3_plus"] >= config.min_repeat_learners
            and row["positive_gap_count"] >= config.min_positive_gaps
            and row["item_accuracy_p90_p10_spread"] >= config.min_item_accuracy_spread
            and config.min_skill_accuracy <= row["accuracy"] <= config.max_skill_accuracy
        )
        relaxed_ok = (
            row["eligible_items"] >= config.min_items_per_skill
            and row["unique_learners"] >= config.relaxed_min_unique_learners
            and row["learners_3_plus"] >= config.relaxed_min_repeat_learners
            and row["positive_gap_count"] >= config.relaxed_min_positive_gaps
            and row["item_accuracy_p90_p10_spread"] >= config.relaxed_min_item_accuracy_spread
            and config.min_skill_accuracy <= row["accuracy"] <= config.max_skill_accuracy
        )
        row["strict_eligible"] = bool(strict_ok)
        row["relaxed_eligible"] = bool(relaxed_ok)
        row["selection_tier"] = "strict" if strict_ok else ("relaxed_only" if relaxed_ok else "ineligible")
        if strict_ok:
            strict.append(row)
        if relaxed_ok:
            relaxed.append(row)
    pool: list[dict[str, Any]]
    tier: str
    if len(strict) >= config.reference_skill_count:
        pool, tier = strict, "strict"
    elif len(relaxed) >= config.reference_skill_count:
        pool, tier = relaxed, "relaxed"
    else:
        raise ReferenceBuildError(
            f"Only {len(strict)} strict and {len(relaxed)} relaxed skills are eligible; "
            f"{config.reference_skill_count} are required. Diagnostics were retained."
        )
    scores = selection_scores(pool)
    for row, score in zip(pool, scores):
        row["selection_score"] = float(score)
    for row in rows:
        row.setdefault("selection_score", math.nan)
        row["selected"] = False
        row["selection_rank"] = None
    ranked = sorted(pool, key=lambda r: (-float(r["selection_score"]), int(r["skill_id"])))
    selected_rows = ranked[: config.reference_skill_count]
    for rank, row in enumerate(selected_rows, start=1):
        row["selected"] = True
        row["selection_rank"] = rank
    return [int(row["skill_id"]) for row in selected_rows], tier


def gap_sample_quantiles(
    selected_skills: Sequence[int],
    gap_samples: Mapping[int, tuple[np.ndarray, np.ndarray]],
    config: BuildConfig,
) -> list[dict[str, Any]]:
    probabilities = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
    rows: list[dict[str, Any]] = []
    for skill in selected_skills:
        values, priorities = gap_samples[int(skill)]
        chosen = top_k_indices(priorities, config.gap_sample_final_per_skill)
        sample = values[chosen].astype(np.float64)
        for probability in probabilities:
            rows.append({
                "skill_id": int(skill),
                "probability": probability,
                "gap_ms": float(np.quantile(sample, probability)) if sample.size else None,
                "sample_size": int(sample.size),
                "method": "deterministic_min_hash_sample",
            })
    return rows


def quota_for(split_code: int, class_code: int, config: BuildConfig) -> int:
    if class_code == 0:
        return config.cold_quota_per_split
    if split_code == 0:
        return config.repeat_train_quota
    if split_code == 1:
        return config.repeat_calibration_quota
    return config.repeat_test_quota


def update_topk_bucket(
    existing: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    learner: np.ndarray,
    skill: np.ndarray,
    event_count: np.ndarray,
    priority: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if existing is not None:
        learner = np.concatenate([existing[0], learner])
        skill = np.concatenate([existing[1], skill])
        event_count = np.concatenate([existing[2], event_count])
        priority = np.concatenate([existing[3], priority])
    chosen = top_k_indices(priority, k)
    return learner[chosen], skill[chosen], event_count[chosen], priority[chosen]


def select_learner_skill_pairs(
    shards: Sequence[SourceShard],
    work_dir: Path,
    selected_skills: Sequence[int],
    config: BuildConfig,
) -> dict[str, np.ndarray]:
    _, _, pq = require_pyarrow()
    selected_set = set(int(x) for x in selected_skills)
    buckets: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for shard in shards:
        table = pq.read_table(pass2_group_path(work_dir, shard.shard_id))
        learner = safe_numpy(table.column("learner_id"), -1, np.int32)
        skill = safe_numpy(table.column("skill_id"), -1, np.int32)
        split = safe_numpy(table.column("split_code"), -1, np.int8)
        count = safe_numpy(table.column("event_count"), 0, np.int32)
        selected_mask = np.isin(skill, np.array(selected_skills, dtype=np.int32))
        learner = learner[selected_mask]
        skill = skill[selected_mask]
        split = split[selected_mask]
        count = count[selected_mask]
        class_code = (count >= 3).astype(np.int8)
        bucket_code = skill.astype(np.int64) * 10 + split.astype(np.int64) * 2 + class_code.astype(np.int64)
        for code in np.unique(bucket_code):
            local = bucket_code == code
            skill_value = int(skill[local][0])
            split_value = int(split[local][0])
            class_value = int(class_code[local][0])
            if skill_value not in selected_set:
                continue
            priority = sample_priorities(learner[local], skill[local], class_code[local])
            key = (skill_value, split_value, class_value)
            buckets[key] = update_topk_bucket(
                buckets.get(key), learner[local], skill[local], count[local], priority,
                quota_for(split_value, class_value, config),
            )
    output: dict[str, list[np.ndarray]] = {
        "learner_id": [], "skill_id": [], "split_code": [], "class_code": [],
        "event_count": [], "selection_priority": [],
    }
    for skill in selected_skills:
        for split_code in range(3):
            for class_code in range(2):
                key = (int(skill), split_code, class_code)
                if key not in buckets:
                    continue
                learner, skill_arr, count, priority = buckets[key]
                output["learner_id"].append(learner)
                output["skill_id"].append(skill_arr)
                output["split_code"].append(np.full(learner.size, split_code, dtype=np.int8))
                output["class_code"].append(np.full(learner.size, class_code, dtype=np.int8))
                output["event_count"].append(count)
                output["selection_priority"].append(priority)
    result = {
        name: np.concatenate(parts) if parts else np.empty(0, dtype=(np.uint64 if name == "selection_priority" else np.int32))
        for name, parts in output.items()
    }
    order = np.lexsort((result["learner_id"], result["class_code"], result["split_code"], result["skill_id"]))
    for name in result:
        result[name] = result[name][order]
    return result


def write_selected_pairs(pairs: Mapping[str, np.ndarray], path: Path, config: BuildConfig) -> None:
    pa, _, _ = require_pyarrow()
    split_names = [SPLIT_CODE_TO_NAME[int(x)] for x in pairs["split_code"]]
    class_names = [SAMPLE_CLASS_NAMES[int(x)] for x in pairs["class_code"]]
    table = pa.table({
        "learner_id": pa.array(pairs["learner_id"], type=pa.int32()),
        "skill_id": pa.array(pairs["skill_id"], type=pa.int32()),
        "split": pa.array(split_names, type=pa.string()),
        "sample_class": pa.array(class_names, type=pa.string()),
        "event_count": pa.array(pairs["event_count"], type=pa.int32()),
        "selection_priority_u64": pa.array(pairs["selection_priority"], type=pa.uint64()),
    })
    write_parquet_atomic(table, path, config.compression, config.compression_level)


def selected_pair_lookup(pairs: Mapping[str, np.ndarray], max_skill_id: int) -> dict[str, np.ndarray]:
    keys = pairs["learner_id"].astype(np.int64) * np.int64(max_skill_id + 1) + pairs["skill_id"].astype(np.int64)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    if sorted_keys.size and np.any(sorted_keys[1:] == sorted_keys[:-1]):
        raise ReferenceBuildError("Duplicate selected learner-skill pair")
    return {
        "keys": sorted_keys,
        "split_code": pairs["split_code"][order],
        "class_code": pairs["class_code"][order],
    }


def reference_output_schema() -> Any:
    pa, _, _ = require_pyarrow()
    return pa.schema([
        ("learner_id", pa.int32()),
        ("skill_id", pa.int32()),
        ("split", pa.string()),
        ("sample_class", pa.string()),
        ("within_skill_index", pa.int32()),
        ("timestamp_ms", pa.int64()),
        ("gap_ms", pa.int64()),
        ("sequence_index", pa.int32()),
        ("question_id", pa.string()),
        ("is_correct", pa.bool_()),
        ("elapsed_time_ms", pa.int64()),
        ("part", pa.int8()),
        ("quality_flags", pa.uint16()),
    ])


def process_pass3_shard(
    shard: SourceShard,
    selected_item_mask: np.ndarray,
    pair_lookup: Mapping[str, np.ndarray],
    max_skill_id: int,
    output_path: Path,
    config: BuildConfig,
) -> None:
    pa, pc, pq = require_pyarrow()
    source_columns = [
        "learner_id", "sequence_index", "timestamp_ms", "question_id", "is_correct",
        "elapsed_time_ms", "part", "primary_skill_id", "skill_count", "quality_flags",
    ]
    table = pq.read_table(shard.interaction_path, columns=source_columns)
    arrays = filter_candidate_arrays(table, selected_item_mask)
    learner = arrays["learner_id"]
    skill = arrays["skill_id"]
    if learner.size == 0 or pair_lookup["keys"].size == 0:
        empty = pa.Table.from_arrays([pa.array([], type=f.type) for f in reference_output_schema()], schema=reference_output_schema())
        write_parquet_atomic(empty, output_path, config.compression, config.compression_level)
        return
    keys = learner.astype(np.int64) * np.int64(max_skill_id + 1) + skill.astype(np.int64)
    positions = np.searchsorted(pair_lookup["keys"], keys)
    matched = positions < pair_lookup["keys"].size
    safe_positions = np.minimum(positions, max(0, pair_lookup["keys"].size - 1))
    matched &= pair_lookup["keys"][safe_positions] == keys
    if not np.any(matched):
        empty = pa.Table.from_arrays([pa.array([], type=f.type) for f in reference_output_schema()], schema=reference_output_schema())
        write_parquet_atomic(empty, output_path, config.compression, config.compression_level)
        return
    selected_source_indices = arrays["source_indices"][matched]
    learner = learner[matched]
    skill = skill[matched]
    timestamp = arrays["timestamp_ms"][matched]
    sequence = arrays["sequence_index"][matched]
    lookup_pos = positions[matched]
    split_code = pair_lookup["split_code"][lookup_pos]
    class_code = pair_lookup["class_code"][lookup_pos]
    order = np.lexsort((sequence, timestamp, skill, learner))
    learner = learner[order]
    skill = skill[order]
    timestamp = timestamp[order]
    sequence = sequence[order]
    split_code = split_code[order]
    class_code = class_code[order]
    selected_source_indices = selected_source_indices[order]
    group_start = np.ones(learner.size, dtype=np.bool_)
    group_start[1:] = (learner[1:] != learner[:-1]) | (skill[1:] != skill[:-1])
    starts = np.flatnonzero(group_start)
    within = np.arange(learner.size, dtype=np.int32)
    repeated_starts = np.repeat(starts, np.diff(np.r_[starts, learner.size]))
    within -= repeated_starts.astype(np.int32)
    gaps = np.zeros(learner.size, dtype=np.int64)
    same = ~group_start
    gaps[same] = timestamp[same] - timestamp[np.flatnonzero(same) - 1]
    source_indices_arrow = pa.array(selected_source_indices, type=pa.int64())
    question = pc.take(table.column("question_id").combine_chunks(), source_indices_arrow)
    correctness = pc.take(table.column("is_correct").combine_chunks(), source_indices_arrow)
    elapsed = pc.take(table.column("elapsed_time_ms").combine_chunks(), source_indices_arrow)
    part = pc.take(table.column("part").combine_chunks(), source_indices_arrow)
    flags = pc.take(table.column("quality_flags").combine_chunks(), source_indices_arrow)
    out = pa.Table.from_arrays([
        pa.array(learner, type=pa.int32()),
        pa.array(skill, type=pa.int32()),
        pa.array([SPLIT_CODE_TO_NAME[int(x)] for x in split_code], type=pa.string()),
        pa.array([SAMPLE_CLASS_NAMES[int(x)] for x in class_code], type=pa.string()),
        pa.array(within, type=pa.int32()),
        pa.array(timestamp, type=pa.int64()),
        pa.array(gaps, type=pa.int64()),
        pa.array(sequence, type=pa.int32()),
        question,
        correctness,
        elapsed,
        part,
        flags,
    ], schema=reference_output_schema())
    write_parquet_atomic(out, output_path, config.compression, config.compression_level)


def merge_reference_parts(shards: Sequence[SourceShard], work_dir: Path, output_path: Path, config: BuildConfig) -> int:
    _, _, pq = require_pyarrow()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".tmp")
    writer = pq.ParquetWriter(
        tmp,
        reference_output_schema(),
        compression=config.compression,
        compression_level=config.compression_level,
        use_dictionary=True,
        write_statistics=True,
    )
    total = 0
    try:
        for shard in shards:
            path = pass3_path(work_dir, shard.shard_id)
            if not path.exists():
                raise ReferenceBuildError(f"Missing pass-3 checkpoint: {path}")
            table = pq.read_table(path)
            if table.schema != reference_output_schema():
                table = table.cast(reference_output_schema())
            writer.write_table(table)
            total += table.num_rows
    finally:
        writer.close()
    row_count, _ = parquet_file_info(tmp)
    if row_count != total:
        tmp.unlink(missing_ok=True)
        raise ReferenceBuildError("Merged reference Parquet verification failed")
    os.replace(tmp, output_path)
    return total


def validate_reference_dataset(
    path: Path,
    pairs: Mapping[str, np.ndarray],
    max_skill_id: int,
) -> dict[str, Any]:
    _, _, pq = require_pyarrow()
    table = pq.read_table(
        path,
        columns=[
            "learner_id", "skill_id", "split", "sample_class",
            "within_skill_index", "gap_ms", "quality_flags",
        ],
    )
    expected_rows = int(np.sum(pairs["event_count"], dtype=np.int64))
    if table.num_rows != expected_rows:
        raise ReferenceBuildError(
            f"Reference row count {table.num_rows} does not equal selected-pair event sum {expected_rows}"
        )
    learner = safe_numpy(table.column("learner_id"), -1, np.int32)
    skill = safe_numpy(table.column("skill_id"), -1, np.int32)
    within = safe_numpy(table.column("within_skill_index"), -1, np.int32)
    gaps = safe_numpy(table.column("gap_ms"), -1, np.int64)
    flags = safe_numpy(table.column("quality_flags"), 0, np.uint16)
    if np.any((flags & np.uint16(REFERENCE_EXCLUDE_MASK)) != 0):
        raise ReferenceBuildError("Reference data contain an excluded quality flag")
    if np.any(gaps < 0):
        raise ReferenceBuildError("Reference data contain a negative within-skill gap")
    keys = learner.astype(np.int64) * np.int64(max_skill_id + 1) + skill.astype(np.int64)
    order = np.lexsort((within, keys))
    keys_sorted = keys[order]
    within_sorted = within[order]
    gaps_sorted = gaps[order]
    starts = np.r_[0, np.flatnonzero(keys_sorted[1:] != keys_sorted[:-1]) + 1]
    ends = np.r_[starts[1:], keys_sorted.size]
    observed_keys = keys_sorted[starts]
    observed_counts = (ends - starts).astype(np.int32)
    expected_keys = pairs["learner_id"].astype(np.int64) * np.int64(max_skill_id + 1) + pairs["skill_id"].astype(np.int64)
    expected_order = np.argsort(expected_keys, kind="stable")
    if not np.array_equal(observed_keys, expected_keys[expected_order]):
        raise ReferenceBuildError("Reference learner-skill keys differ from the selected pair table")
    if not np.array_equal(observed_counts, pairs["event_count"][expected_order]):
        raise ReferenceBuildError("Reference per-pair event counts differ from the selected pair table")
    for start, end in zip(starts, ends):
        expected_index = np.arange(end - start, dtype=np.int32)
        if not np.array_equal(within_sorted[start:end], expected_index):
            raise ReferenceBuildError("within_skill_index is not contiguous from zero")
        if gaps_sorted[start] != 0:
            raise ReferenceBuildError("The first event of a learner-skill sequence must have gap_ms=0")
    split_values = table.column("split").to_pylist()
    class_values = table.column("sample_class").to_pylist()
    expected_split_by_key = {
        int(key): SPLIT_CODE_TO_NAME[int(code)]
        for key, code in zip(expected_keys, pairs["split_code"])
    }
    expected_class_by_key = {
        int(key): SAMPLE_CLASS_NAMES[int(code)]
        for key, code in zip(expected_keys, pairs["class_code"])
    }
    for key, split_name, class_name in zip(keys, split_values, class_values):
        if split_name != expected_split_by_key[int(key)]:
            raise ReferenceBuildError("Reference split label differs from selected pair metadata")
        if class_name != expected_class_by_key[int(key)]:
            raise ReferenceBuildError("Reference sample_class differs from selected pair metadata")
    return {
        "status": "PASS",
        "rows": int(table.num_rows),
        "learner_skill_pairs": int(observed_keys.size),
        "excluded_flag_rows": 0,
        "negative_gap_rows": 0,
    }


def build_learner_splits(shards: Sequence[SourceShard], output_path: Path, config: BuildConfig) -> dict[str, int]:
    pa, _, pq = require_pyarrow()
    schema = pa.schema([("learner_id", pa.int32()), ("split", pa.string())])
    tmp = output_path.with_name(output_path.name + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(
        tmp, schema, compression=config.compression, compression_level=config.compression_level,
        use_dictionary=True, write_statistics=True,
    )
    counts = {name: 0 for name in SPLIT_NAMES}
    total = 0
    seen_min: int | None = None
    seen_max: int | None = None
    try:
        for shard in shards:
            table = pq.read_table(shard.learner_path, columns=["learner_id"])
            learner = safe_numpy(table.column("learner_id"), -1, np.int32)
            split = learner_split_codes(learner)
            names = [SPLIT_CODE_TO_NAME[int(x)] for x in split]
            writer.write_table(pa.table({
                "learner_id": pa.array(learner, type=pa.int32()),
                "split": pa.array(names, type=pa.string()),
            }, schema=schema))
            total += learner.size
            for code, name in enumerate(SPLIT_NAMES):
                counts[name] += int(np.sum(split == code))
            if learner.size:
                seen_min = int(learner.min()) if seen_min is None else min(seen_min, int(learner.min()))
                seen_max = int(learner.max()) if seen_max is None else max(seen_max, int(learner.max()))
    finally:
        writer.close()
    row_count, _ = parquet_file_info(tmp)
    if row_count != total:
        tmp.unlink(missing_ok=True)
        raise ReferenceBuildError("learner_splits.parquet verification failed")
    os.replace(tmp, output_path)
    counts["total"] = total
    counts["minimum_learner_id"] = seen_min if seen_min is not None else -1
    counts["maximum_learner_id"] = seen_max if seen_max is not None else -1
    return counts


def zip_final_bundle(final_dir: Path, zip_path: Path) -> None:
    included = [p for p in sorted(final_dir.rglob("*")) if p.is_file() and p != zip_path]
    tmp = zip_path.with_name(zip_path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as zf:
        for path in included:
            rel = path.relative_to(final_dir).as_posix()
            compression = zipfile.ZIP_STORED if path.suffix.lower() == ".parquet" else zipfile.ZIP_DEFLATED
            zf.write(path, rel, compress_type=compression)
    with zipfile.ZipFile(tmp, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            tmp.unlink(missing_ok=True)
            raise ReferenceBuildError(f"ZIP integrity failure at {bad}")
    os.replace(tmp, zip_path)


def write_sha256s(final_dir: Path, output_path: Path) -> None:
    rows = []
    for path in sorted(final_dir.rglob("*")):
        if path.is_file() and path != output_path and path.suffix.lower() != ".zip":
            rows.append(f"{sha256_file(path)}  {path.relative_to(final_dir).as_posix()}")
    atomic_write_text(output_path, "\n".join(rows) + "\n")


def preflight(input_dir: Path, output_dir: Path, shards: Sequence[SourceShard]) -> dict[str, Any]:
    pa, _, pq = require_pyarrow()
    free = shutil.disk_usage(output_dir.parent if output_dir.parent.exists() else input_dir).free
    source_size = sum(s.interaction_path.stat().st_size + s.learner_path.stat().st_size for s in shards)
    required = max(2 * 1024**3, int(source_size * 0.45))
    if free < required:
        raise ReferenceBuildError(
            f"Insufficient free disk space. Available {human_bytes(free)}, "
            f"required safety minimum {human_bytes(required)}."
        )
    return {
        "status": "PASS",
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pyarrow_version": pa.__version__,
        "source_shards": len(shards),
        "source_interaction_rows": int(sum(s.interaction_rows for s in shards)),
        "source_learner_rows": int(sum(s.learner_rows for s in shards)),
        "source_parquet_bytes": source_size,
        "disk_free_bytes": free,
        "disk_required_safety_bytes": required,
        "parquet_smoke_schema": str(parquet_file_info(shards[0].interaction_path)[1]),
    }


def write_diagnostics_early(final_dir: Path, stats: Mapping[int, Mapping[str, Any]]) -> None:
    rows = [dict(stats[key]) for key in sorted(stats)]
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    write_csv(final_dir / "skill_selection_diagnostics.csv", fields, rows)


def run_build(input_dir: Path, output_dir: Path, config: BuildConfig) -> Path:
    started = time.monotonic()
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / ".work"
    final_dir = output_dir / "final"
    work_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    source_manifest, shards = load_source(input_dir)
    question_meta = load_question_metadata(input_dir)
    source_manifest_path = input_dir / "manifest.json"
    source_manifest_sha = sha256_file(source_manifest_path)
    identity = {
        "source_manifest_sha256": source_manifest_sha,
        "config": asdict(config),
    }
    identity_hash = sha256_text(canonical_json(identity))
    identity_path = output_dir / "run_identity.json"
    if identity_path.exists():
        previous = load_json(identity_path)
        if previous.get("identity_hash") != identity_hash:
            raise ReferenceBuildError(
                "OutputDir belongs to a different source or configuration. "
                "Choose a new OutputDir instead of mixing runs."
            )
    else:
        atomic_write_json(identity_path, {**identity, "identity_hash": identity_hash, "created_at_utc": utc_now_iso()})

    if (output_dir / "COMPLETE.json").exists():
        complete = load_json(output_dir / "COMPLETE.json")
        zip_path = Path(complete["bundle_path"])
        if zip_path.exists() and sha256_file(zip_path) == complete["bundle_sha256"]:
            print(f"BUILD: ALREADY COMPLETE\nBUNDLE: {zip_path}")
            return zip_path
        raise ReferenceBuildError("COMPLETE marker exists but final bundle is missing or corrupted")

    report = preflight(input_dir, output_dir, shards)
    atomic_write_json(output_dir / "preflight_report.json", report)
    print(f"PREFLIGHT: PASS | shards={len(shards)} | rows={report['source_interaction_rows']:,}")

    # PASS 1: exact item counts.
    for number, shard in enumerate(shards, start=1):
        path = pass1_partial_path(work_dir, shard.shard_id)
        if not pass1_checkpoint_valid(path, shard, question_meta["qmax"]):
            path.unlink(missing_ok=True)
            process_pass1_shard(shard, question_meta["qmax"], path)
        if number % 25 == 0 or number == len(shards):
            print(f"PASS 1/3: {number}/{len(shards)} shards")
    item_rows, item_correct, retained_pass1 = merge_pass1(shards, work_dir, question_meta["qmax"])
    eligible_item_mask, candidate_item_mask, candidate_skills = candidate_items(
        item_rows, item_correct, question_meta, config
    )
    atomic_write_json(output_dir / "candidate_definition.json", {
        "candidate_skills": candidate_skills,
        "candidate_skill_count": len(candidate_skills),
        "eligible_item_count": int(eligible_item_mask.sum()),
        "candidate_item_count": int(candidate_item_mask.sum()),
        "pass1_retained_rows": retained_pass1,
        "reference_exclude_mask": REFERENCE_EXCLUDE_MASK,
    })
    print(f"ITEM SCREEN: {len(candidate_skills)} candidate skills, {int(candidate_item_mask.sum()):,} candidate items")

    # PASS 2: learner-skill and gap summaries.
    for number, shard in enumerate(shards, start=1):
        group_path = pass2_group_path(work_dir, shard.shard_id)
        gap_path = pass2_gap_path(work_dir, shard.shard_id)
        if not pass2_checkpoint_valid(group_path, gap_path, shard, candidate_skills):
            group_path.unlink(missing_ok=True)
            gap_path.unlink(missing_ok=True)
            process_pass2_shard(
                shard, candidate_item_mask, candidate_skills, group_path, gap_path, config
            )
        if number % 25 == 0 or number == len(shards):
            print(f"PASS 2/3: {number}/{len(shards)} shards")
    stats, gap_hist, gap_samples = aggregate_pass2(shards, work_dir, candidate_skills)
    item_rows_output = add_item_distribution_stats(
        stats, eligible_item_mask, candidate_item_mask, item_rows, item_correct, question_meta
    )
    try:
        selected_skills, selection_tier = choose_reference_skills(stats, config)
    except Exception:
        write_diagnostics_early(final_dir, stats)
        raise
    print(f"SKILL SELECTION: {selection_tier} tier | selected={selected_skills}")

    selected_item_mask = candidate_item_mask & np.isin(
        question_meta["q_to_skill"], np.array(selected_skills, dtype=np.int32)
    )
    selected_items = [row for row in item_rows_output if row["skill_id"] in set(selected_skills)]

    # Select deterministic learner-skill sample and write it before pass 3.
    pairs = select_learner_skill_pairs(shards, work_dir, selected_skills, config)
    selected_pairs_path = final_dir / "selected_learner_skill_pairs.parquet"
    write_selected_pairs(pairs, selected_pairs_path, config)
    pair_lookup = selected_pair_lookup(pairs, question_meta["max_skill_id"])
    print(f"PAIR SAMPLE: {pairs['learner_id'].size:,} learner-skill pairs")

    # PASS 3: extract complete selected-item sequences for sampled pairs.
    for number, shard in enumerate(shards, start=1):
        path = pass3_path(work_dir, shard.shard_id)
        if not pass3_checkpoint_valid(path):
            path.unlink(missing_ok=True)
            process_pass3_shard(
                shard, selected_item_mask, pair_lookup, question_meta["max_skill_id"], path, config
            )
        if number % 25 == 0 or number == len(shards):
            print(f"PASS 3/3: {number}/{len(shards)} shards")
    reference_path = final_dir / "reference_interactions.parquet"
    reference_rows = merge_reference_parts(
        shards, work_dir, reference_path, config
    )
    reference_validation = validate_reference_dataset(
        reference_path, pairs, question_meta["max_skill_id"]
    )
    atomic_write_json(final_dir / "reference_validation.json", reference_validation)
    split_counts = build_learner_splits(shards, final_dir / "learner_splits.parquet", config)
    if split_counts["total"] != int(sum(s.learner_rows for s in shards)):
        raise ReferenceBuildError("Learner split total differs from the source manifest")

    # Exact and sampled summary outputs.
    diagnostic_rows = [stats[key] for key in sorted(stats)]
    diagnostic_fields = sorted({key for row in diagnostic_rows for key in row})
    write_csv(final_dir / "skill_selection_diagnostics.csv", diagnostic_fields, diagnostic_rows)
    selected_rows = [row for row in diagnostic_rows if row["selected"]]
    selected_rows.sort(key=lambda row: int(row["selection_rank"]))
    write_csv(final_dir / "selected_skills.csv", diagnostic_fields, selected_rows)
    item_fields = list(item_rows_output[0].keys()) if item_rows_output else []
    write_csv(final_dir / "candidate_items.csv", item_fields, item_rows_output)
    write_csv(final_dir / "selected_items.csv", item_fields, selected_items)

    gap_hist_rows: list[dict[str, Any]] = []
    candidate_index = {int(s): i for i, s in enumerate(candidate_skills)}
    for skill in selected_skills:
        i = candidate_index[int(skill)]
        for b, label in enumerate(GAP_BIN_LABELS):
            gap_hist_rows.append({
                "skill_id": int(skill),
                "gap_bin": label,
                "positive_gap_rows": int(gap_hist[i, b]),
            })
    write_csv(final_dir / "selected_skill_gap_histogram.csv", ["skill_id", "gap_bin", "positive_gap_rows"], gap_hist_rows)
    gap_quantile_rows = gap_sample_quantiles(selected_skills, gap_samples, config)
    write_csv(
        final_dir / "selected_skill_gap_sample_quantiles.csv",
        ["skill_id", "probability", "gap_ms", "sample_size", "method"],
        gap_quantile_rows,
    )

    selected_pair_summary: list[dict[str, Any]] = []
    for skill in selected_skills:
        for split_code in range(3):
            for class_code in range(2):
                mask = (
                    (pairs["skill_id"] == skill)
                    & (pairs["split_code"] == split_code)
                    & (pairs["class_code"] == class_code)
                )
                selected_pair_summary.append({
                    "skill_id": int(skill),
                    "split": SPLIT_CODE_TO_NAME[split_code],
                    "sample_class": SAMPLE_CLASS_NAMES[class_code],
                    "selected_pairs": int(mask.sum()),
                    "source_events_for_pairs": int(pairs["event_count"][mask].sum()),
                    "quota": quota_for(split_code, class_code, config),
                })
    write_csv(
        final_dir / "selected_pair_summary.csv",
        ["skill_id", "split", "sample_class", "selected_pairs", "source_events_for_pairs", "quota"],
        selected_pair_summary,
    )

    # Copy question metadata needed to interpret the compact data.
    _, _, pq = require_pyarrow()
    question_table = question_meta["table"]
    qnums = np.array([int(str(q)[1:]) for q in question_table.column("question_id").to_pylist()], dtype=np.int32)
    selected_question_rows = selected_item_mask[qnums]
    write_parquet_atomic(
        question_table.filter(require_pyarrow()[0].array(selected_question_rows)),
        final_dir / "selected_questions.parquet",
        config.compression,
        config.compression_level,
    )

    manifest = {
        "created_at_utc": utc_now_iso(),
        "source": {
            "input_directory": str(input_dir),
            "source_manifest_sha256": source_manifest_sha,
            "source_interaction_rows": int(sum(s.interaction_rows for s in shards)),
            "source_learner_rows": int(sum(s.learner_rows for s in shards)),
            "source_shards": len(shards),
        },
        "filter": {
            "single_skill_only": True,
            "require_valid_correctness": True,
            "require_valid_timestamp": True,
            "excluded_quality_mask": REFERENCE_EXCLUDE_MASK,
            "excluded_quality_bits": [
                "UNKNOWN_QUESTION", "MISSING_CORRECT_ANSWER", "MISSING_SKILL_TAG",
                "DUPLICATE_EVENT", "INVALID_RESPONSE", "INVALID_TIMESTAMP",
            ],
            "question_before_deployment_excluded": False,
        },
        "selection": {
            "selection_tier": selection_tier,
            "selected_skills": selected_skills,
            "selected_skill_count": len(selected_skills),
            "selected_item_count": int(selected_item_mask.sum()),
            "candidate_skill_count": len(candidate_skills),
            "candidate_item_count": int(candidate_item_mask.sum()),
        },
        "splits": {
            "method": "splitmix64(learner_id XOR fixed_salt) modulo 10000",
            "thresholds": {"train": "0-6999", "calibration": "7000-8499", "test": "8500-9999"},
            "counts": split_counts,
        },
        "sample": {
            "method": "smallest deterministic hash priorities within skill/split/class",
            "selected_learner_skill_pairs": int(pairs["learner_id"].size),
            "reference_interaction_rows": int(reference_rows),
            "quotas": {
                "repeat_train": config.repeat_train_quota,
                "repeat_calibration": config.repeat_calibration_quota,
                "repeat_test": config.repeat_test_quota,
                "cold_each_split": config.cold_quota_per_split,
            },
        },
        "config": asdict(config),
        "run_identity_hash": identity_hash,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_write_json(final_dir / "reference_manifest.json", manifest)
    readme = f"""# AdaptiveLearningSim EdNet-KT1 reference bundle

This bundle was produced deterministically from the complete local EdNet-KT1
Parquet output.

- Selected skills: {', '.join(str(x) for x in selected_skills)}
- Selected items: {int(selected_item_mask.sum()):,}
- Sampled learner-skill pairs: {int(pairs['learner_id'].size):,}
- Reference interactions: {reference_rows:,}
- Split rule: learner-level 70% train, 15% calibration, 15% test
- Selection tier: {selection_tier}

`QUESTION_BEFORE_DEPLOYMENT` was retained and was not used as an exclusion,
because privacy-shifted KT timestamps are not proven comparable with item
deployment timestamps.
"""
    atomic_write_text(final_dir / "README.md", readme)
    write_sha256s(final_dir, final_dir / "SHA256SUMS.txt")
    zip_path = output_dir / "AdaptiveLearningSim_KT1_reference_bundle.zip"
    zip_final_bundle(final_dir, zip_path)
    complete = {
        "status": "PASS",
        "completed_at_utc": utc_now_iso(),
        "bundle_path": str(zip_path),
        "bundle_sha256": sha256_file(zip_path),
        "bundle_bytes": zip_path.stat().st_size,
        "reference_interaction_rows": reference_rows,
        "selected_skills": selected_skills,
    }
    atomic_write_json(output_dir / "COMPLETE.json", complete)
    print("Reference build completed successfully.")
    print(f"BUNDLE: {zip_path}")
    print(f"BUNDLE SIZE: {human_bytes(zip_path.stat().st_size)}")
    return zip_path


def synthetic_source(root: Path) -> tuple[Path, BuildConfig]:
    pa, _, pq = require_pyarrow()
    source = root / "source"
    (source / "interactions").mkdir(parents=True)
    (source / "learners").mkdir(parents=True)
    (source / "metadata").mkdir(parents=True)
    rng = np.random.default_rng(20260731)
    questions: list[dict[str, Any]] = []
    qnum = 1
    skill_items: dict[int, list[str]] = {}
    for skill in range(100, 122):
        skill_items[skill] = []
        for item_offset in range(3):
            qid = f"q{qnum}"
            skill_items[skill].append(qid)
            questions.append({
                "question_id": qid,
                "bundle_id": f"b{qnum}",
                "explanation_id": None,
                "correct_answer": "a",
                "part": 1 + (skill % 3),
                "skill_ids": str(skill),
                "skill_count": 1,
                "primary_skill_id": skill,
                "deployed_at_ms": None,
            })
            qnum += 1
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
    qtable = pa.Table.from_pylist(questions, schema=question_schema)
    pq.write_table(qtable, source / "metadata" / "questions.parquet", compression="zstd")
    shards_manifest: list[dict[str, Any]] = []
    learner_ids = np.arange(1, 61, dtype=np.int32)
    for shard_id, learner_chunk in enumerate(np.array_split(learner_ids, 2)):
        rows: list[dict[str, Any]] = []
        learner_rows: list[dict[str, Any]] = []
        for learner_id in learner_chunk:
            seq = 0
            timestamp = int(learner_id) * 1_000_000
            learner_count = 0
            for skill in range(100, 122):
                repeats = 3 + ((int(learner_id) + skill) % 3)
                for rep in range(repeats):
                    qid = skill_items[skill][rep % 3]
                    # Skill-specific and item-specific variation ensures spread.
                    probability = 0.25 + 0.02 * (skill - 100) + 0.15 * (rep % 3)
                    is_correct = bool(rng.random() < min(0.95, probability))
                    rows.append({
                        "learner_id": int(learner_id),
                        "sequence_index": seq,
                        "timestamp_ms": timestamp,
                        "solving_id": seq + 1,
                        "question_id": qid,
                        "bundle_id": qid.replace("q", "b"),
                        "user_answer": "a" if is_correct else "b",
                        "correct_answer": "a",
                        "is_correct": is_correct,
                        "elapsed_time_ms": 5_000 + rep * 100,
                        "part": 1 + (skill % 3),
                        "skill_ids": str(skill),
                        "skill_count": 1,
                        "primary_skill_id": skill,
                        "deployed_at_ms": None,
                        "source_file": f"u{learner_id}.csv",
                        "source_row": seq + 2,
                        "quality_flags": 0,
                    })
                    seq += 1
                    learner_count += 1
                    timestamp += 86_400_000 + rep * 1_000
            learner_rows.append({"learner_id": int(learner_id)})
        interaction_path = source / "interactions" / f"part-{shard_id:06d}.parquet"
        learner_path = source / "learners" / f"part-{shard_id:06d}.parquet"
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
        learner_schema = pa.schema([("learner_id", pa.int32())])
        pq.write_table(pa.Table.from_pylist(rows, schema=interaction_schema), interaction_path, compression="zstd")
        pq.write_table(pa.Table.from_pylist(learner_rows, schema=learner_schema), learner_path, compression="zstd")
        shards_manifest.append({
            "shard_id": shard_id,
            "interaction_output": str(interaction_path.relative_to(source)),
            "learner_output": str(learner_path.relative_to(source)),
            "interaction_rows": len(rows),
            "learner_rows": len(learner_rows),
            "interaction_sha256": sha256_file(interaction_path),
            "learner_sha256": sha256_file(learner_path),
        })
    atomic_write_json(source / "manifest.json", {
        "output": {"interaction_rows": sum(x["interaction_rows"] for x in shards_manifest), "learner_rows": 60},
        "shards": shards_manifest,
    })
    config = BuildConfig(
        reference_skill_count=20,
        min_item_interactions=20,
        min_items_per_skill=2,
        min_unique_learners=20,
        min_repeat_learners=20,
        min_positive_gaps=20,
        min_item_accuracy_spread=0.0,
        min_skill_accuracy=0.0,
        max_skill_accuracy=1.0,
        relaxed_min_unique_learners=10,
        relaxed_min_repeat_learners=10,
        relaxed_min_positive_gaps=10,
        relaxed_min_item_accuracy_spread=0.0,
        repeat_train_quota=10,
        repeat_calibration_quota=5,
        repeat_test_quota=5,
        cold_quota_per_split=2,
        gap_sample_per_shard_skill=50,
        gap_sample_final_per_skill=100,
    )
    return source, config


def remove_tree_with_retries(path: Path, attempts: int = 20, delay_seconds: float = 0.20) -> bool:
    """Best-effort Windows-safe recursive cleanup without masking test results."""
    if not path.exists():
        return True
    for _ in range(attempts):
        gc.collect()
        try:
            shutil.rmtree(path)
            return True
        except FileNotFoundError:
            return True
        except PermissionError:
            time.sleep(delay_seconds)
    return not path.exists()


def run_self_test(self_test_root: Path | None = None) -> None:
    pa, _, pq = require_pyarrow()
    root_parent = None
    if self_test_root is not None:
        self_test_root = self_test_root.resolve()
        self_test_root.mkdir(parents=True, exist_ok=True)
        root_parent = str(self_test_root)
    root = Path(tempfile.mkdtemp(prefix="als_kt1_reference_selftest_", dir=root_parent))
    try:
        source, config = synthetic_source(root)
        output = root / "output"
        bundle = run_build(source, output, config)
        if not bundle.exists():
            raise ReferenceBuildError("Self-test bundle was not created")
        manifest = load_json(output / "final" / "reference_manifest.json")
        if len(manifest["selection"]["selected_skills"]) != 20:
            raise ReferenceBuildError("Self-test did not select exactly 20 skills")
        table = pq.read_table(output / "final" / "reference_interactions.parquet")
        if table.num_rows <= 0:
            raise ReferenceBuildError("Self-test reference dataset is empty")
        flags = safe_numpy(table.column("quality_flags"), 0, np.uint16)
        if np.any((flags & np.uint16(REFERENCE_EXCLUDE_MASK)) != 0):
            raise ReferenceBuildError("Self-test retained excluded quality flags")
        split_table = pq.read_table(output / "final" / "learner_splits.parquet")
        learners = safe_numpy(split_table.column("learner_id"), -1, np.int32)
        if np.unique(learners).size != learners.size:
            raise ReferenceBuildError("Self-test learner split contains duplicate learners")
        with zipfile.ZipFile(bundle, "r") as zf:
            if zf.testzip() is not None:
                raise ReferenceBuildError("Self-test ZIP failed integrity verification")
        del table, split_table, flags, learners, manifest
        gc.collect()
    finally:
        if not remove_tree_with_retries(root):
            print(
                f"WARNING: self-test passed but Windows kept a temporary file locked; "
                f"the small directory can be deleted later: {root}",
                file=sys.stderr,
            )
    print(f"SELF-TEST: PASS | pyarrow={pa.__version__} | numpy={np.__version__}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, help="Completed EdNet-KT1 preprocessing output directory")
    parser.add_argument("--output-dir", type=Path, help="Reference-builder output directory")
    parser.add_argument("--self-test", action="store_true", help="Run full synthetic Parquet self-test")
    parser.add_argument("--self-test-root", type=Path, help="Optional parent directory for self-test files")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.self_test:
            run_self_test(args.self_test_root)
            return 0
        if args.input_dir is None or args.output_dir is None:
            raise ReferenceBuildError("--input-dir and --output-dir are required unless --self-test is used")
        run_build(args.input_dir, args.output_dir, BuildConfig())
        return 0
    except ReferenceBuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: checkpoints are retained; rerun the same command to continue.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"UNEXPECTED ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
