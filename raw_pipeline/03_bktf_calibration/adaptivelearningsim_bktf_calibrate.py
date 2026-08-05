#!/usr/bin/env python3
"""AdaptiveLearningSim EdNet-KT1 BKT/BKT-F calibration pipeline.


The program validates the compact EdNet-KT1 reference bundle, fits a standard
Bayesian Knowledge Tracing model and a time-gap forgetting extension separately
for each selected skill, evaluates both models on learner-disjoint calibration
and test splits, and writes a compact reproducible result bundle.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata

DAY_MS = 86_400_000.0
EPS = 1e-12
MAX_EMISSION_ERROR = 0.49
EXPECTED_FILES = {
    "reference_manifest.json",
    "reference_validation.json",
    "SHA256SUMS.txt",
    "selected_skills.csv",
    "selected_items.csv",
    "selected_pair_summary.csv",
    "learner_splits.parquet",
    "selected_learner_skill_pairs.parquet",
    "selected_questions.parquet",
    "reference_interactions.parquet",
}
EXPECTED_INTERACTION_COLUMNS = [
    "learner_id", "skill_id", "split", "sample_class",
    "within_skill_index", "timestamp_ms", "gap_ms", "sequence_index",
    "question_id", "is_correct", "elapsed_time_ms", "part", "quality_flags",
]
SPLIT_NAMES = ("train", "calibration", "test")
CLASS_NAMES = ("cold_1_2", "repeat_3_plus")
GAP_BINS = [
    ("first_or_zero", -1.0, 0.0),
    ("gt0_lt1min", 0.0, 60_000.0),
    ("1min_lt5min", 60_000.0, 300_000.0),
    ("5min_lt30min", 300_000.0, 1_800_000.0),
    ("30min_lt2h", 1_800_000.0, 7_200_000.0),
    ("2h_lt12h", 7_200_000.0, 43_200_000.0),
    ("12h_lt1d", 43_200_000.0, 86_400_000.0),
    ("1d_lt3d", 86_400_000.0, 259_200_000.0),
    ("3d_lt7d", 259_200_000.0, 604_800_000.0),
    ("7d_lt30d", 604_800_000.0, 2_592_000_000.0),
    ("30d_plus", 2_592_000_000.0, math.inf),
]


class CalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class FitConfig:
    starts_bkt: int = 8
    starts_bktf: int = 12
    maxiter: int = 350
    ftol: float = 1e-11
    gtol: float = 1e-6
    min_calibration_logloss_gain: float = 1e-4
    ece_bins: int = 15
    lambda_min: float = 1e-8
    lambda_max: float = 10.0
    seed: int = 20260731

    def as_dict(self) -> dict[str, Any]:
        return {
            "starts_bkt": self.starts_bkt,
            "starts_bktf": self.starts_bktf,
            "maxiter": self.maxiter,
            "ftol": self.ftol,
            "gtol": self.gtol,
            "min_calibration_logloss_gain": self.min_calibration_logloss_gain,
            "ece_bins": self.ece_bins,
            "lambda_min": self.lambda_min,
            "lambda_max": self.lambda_max,
            "seed": self.seed,
        }


@dataclass
class SequenceBatch:
    lengths: np.ndarray
    starts: np.ndarray
    gaps_days: np.ndarray
    gaps_ms: np.ndarray
    outcomes: np.ndarray
    learner_ids: np.ndarray
    sample_codes: np.ndarray
    skill_id: int
    split_name: str

    @property
    def n_sequences(self) -> int:
        return int(self.lengths.size)

    @property
    def n_events(self) -> int:
        return int(self.outcomes.size)

    @property
    def max_length(self) -> int:
        return int(self.lengths[0]) if self.lengths.size else 0

    def validate(self) -> None:
        if self.n_sequences == 0 or self.n_events == 0:
            raise CalibrationError(f"Empty sequence batch: skill={self.skill_id}, split={self.split_name}")
        if self.lengths.dtype.kind not in "iu":
            raise CalibrationError("Sequence lengths are not integer")
        if np.any(self.lengths <= 0):
            raise CalibrationError("A sequence has non-positive length")
        if np.any(self.lengths[:-1] < self.lengths[1:]):
            raise CalibrationError("Sequence lengths are not sorted descending")
        if int(self.lengths.sum()) != self.n_events:
            raise CalibrationError("Sequence lengths do not sum to event count")
        if self.starts.size != self.lengths.size:
            raise CalibrationError("Starts and lengths differ")
        expected_starts = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(self.lengths[:-1], dtype=np.int64)))
        if not np.array_equal(self.starts, expected_starts):
            raise CalibrationError("Sequence starts are not contiguous")
        if np.any(self.gaps_ms < 0) or np.any(~np.isfinite(self.gaps_days)):
            raise CalibrationError("Invalid time gap")
        if np.any((self.outcomes != 0) & (self.outcomes != 1)):
            raise CalibrationError("Outcome is not binary")
        first_positions = self.starts
        if np.any(self.gaps_ms[first_positions] != 0):
            raise CalibrationError("The first event of a sequence has non-zero gap")
        if np.unique(self.learner_ids).size != self.learner_ids.size:
            raise CalibrationError("Duplicate learner sequence in batch")


@dataclass(frozen=True)
class ModelParameters:
    p_init: float
    p_learn: float
    slip: float
    guess: float
    lambda_per_day: float

    def as_array(self, forgetting: bool) -> np.ndarray:
        if forgetting:
            return np.array([self.p_init, self.p_learn, self.slip, self.guess, self.lambda_per_day], dtype=np.float64)
        return np.array([self.p_init, self.p_learn, self.slip, self.guess], dtype=np.float64)

    def as_dict(self) -> dict[str, float]:
        return {
            "p_init": float(self.p_init),
            "p_learn": float(self.p_learn),
            "slip": float(self.slip),
            "guess": float(self.guess),
            "lambda_per_day": float(self.lambda_per_day),
            "retention_half_life_days": float(math.log(2.0) / self.lambda_per_day) if self.lambda_per_day > 0 else math.inf,
        }


def utc_iso() -> str:
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(json_safe(obj), f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            clean = {}
            for k in fieldnames:
                v = row.get(k)
                if isinstance(v, (np.integer,)):
                    v = int(v)
                elif isinstance(v, (np.floating,)):
                    v = float(v)
                elif isinstance(v, (np.bool_,)):
                    v = bool(v)
                clean[k] = v
            w.writerow(clean)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_sha_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("  ", 1)
            if len(parts) != 2 or len(parts[0]) != 64:
                raise CalibrationError(f"Malformed SHA256SUMS line {line_no}")
            result[parts[1]] = parts[0].lower()
    return result


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            target = (destination / info.filename).resolve()
            if target != root and root not in target.parents:
                raise CalibrationError(f"Unsafe ZIP member: {info.filename}")
        zf.extractall(destination)


def locate_bundle_root(path: Path, temp_parent: Path) -> tuple[Path, Path | None]:
    if path.is_dir():
        return path.resolve(), None
    if not path.is_file() or path.suffix.lower() != ".zip":
        raise CalibrationError("InputBundle must be a directory or ZIP file")
    temp_dir = Path(tempfile.mkdtemp(prefix="als_bktf_input_", dir=temp_parent))
    safe_extract_zip(path, temp_dir)
    candidates = [temp_dir]
    candidates.extend(p for p in temp_dir.rglob("reference_manifest.json") if p.is_file())
    roots: list[Path] = []
    for c in candidates:
        root = c if c.is_dir() else c.parent
        if EXPECTED_FILES.issubset({p.name for p in root.iterdir() if p.is_file()}):
            roots.append(root)
    roots = list(dict.fromkeys(roots))
    if len(roots) != 1:
        raise CalibrationError(f"Could not identify exactly one reference-bundle root; found {len(roots)}")
    return roots[0], temp_dir


def require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise CalibrationError("pyarrow is required; run the supplied PowerShell launcher") from exc
    return pa, pc, pq


def verify_bundle(root: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    files = {p.name for p in root.iterdir() if p.is_file()}
    missing = sorted(EXPECTED_FILES - files)
    if missing:
        raise CalibrationError(f"Reference bundle is missing files: {missing}")
    hashes = parse_sha_file(root / "SHA256SUMS.txt")
    for name, expected in hashes.items():
        p = root / name
        if not p.is_file():
            raise CalibrationError(f"SHA256SUMS references missing file: {name}")
        actual = sha256_file(p)
        if actual.lower() != expected:
            raise CalibrationError(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")
    manifest = read_json(root / "reference_manifest.json")
    validation = read_json(root / "reference_validation.json")
    if validation.get("status") != "PASS":
        raise CalibrationError("Reference validation status is not PASS")
    if int(validation.get("excluded_flag_rows", -1)) != 0:
        raise CalibrationError("Reference data contain excluded quality flags")
    if int(validation.get("negative_gap_rows", -1)) != 0:
        raise CalibrationError("Reference data contain negative gaps")
    if int(manifest.get("selection", {}).get("selected_skill_count", -1)) != 20:
        raise CalibrationError("Expected 20 selected skills")
    if int(manifest.get("sample", {}).get("reference_interaction_rows", -1)) != int(validation.get("rows", -2)):
        raise CalibrationError("Reference row count differs between manifest and validation")
    bundle_identity = hashlib.sha256((root / "reference_manifest.json").read_bytes() + (root / "SHA256SUMS.txt").read_bytes()).hexdigest()
    return manifest, validation, bundle_identity


def parquet_metadata_and_schema(path: Path) -> tuple[int, list[str], int]:
    _, _, pq = require_pyarrow()
    pf = pq.ParquetFile(path)
    try:
        return int(pf.metadata.num_rows), list(pf.schema_arrow.names), int(pf.metadata.num_row_groups)
    finally:
        pf.close(force=True)


def validate_parquet_headers(root: Path, manifest: dict[str, Any], validation: dict[str, Any]) -> None:
    expected = {
        "reference_interactions.parquet": (int(validation["rows"]), EXPECTED_INTERACTION_COLUMNS),
        "selected_learner_skill_pairs.parquet": (int(validation["learner_skill_pairs"]), [
            "learner_id", "skill_id", "split", "sample_class", "event_count", "selection_priority_u64"
        ]),
        "learner_splits.parquet": (int(manifest["splits"]["counts"]["total"]), ["learner_id", "split"]),
        "selected_questions.parquet": (int(manifest["selection"]["selected_item_count"]), [
            "question_id", "bundle_id", "explanation_id", "correct_answer", "part", "skill_ids",
            "skill_count", "primary_skill_id", "deployed_at_ms"
        ]),
    }
    for name, (rows, columns) in expected.items():
        actual_rows, actual_cols, row_groups = parquet_metadata_and_schema(root / name)
        if actual_rows != rows:
            raise CalibrationError(f"Parquet row count mismatch for {name}: {actual_rows} != {rows}")
        if actual_cols != columns:
            raise CalibrationError(f"Parquet schema mismatch for {name}: {actual_cols}")
        if row_groups <= 0:
            raise CalibrationError(f"Parquet file has no row groups: {name}")


def arrow_to_numpy_int(column, dtype) -> np.ndarray:
    arr = column.combine_chunks()
    if arr.null_count:
        raise CalibrationError(f"Unexpected nulls in required column {arr.type}")
    return np.asarray(arr.to_numpy(zero_copy_only=False), dtype=dtype)


def load_reference_arrays(root: Path, manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    pa, pc, pq = require_pyarrow()
    columns = [
        "learner_id", "skill_id", "split", "sample_class", "within_skill_index",
        "gap_ms", "is_correct", "question_id", "quality_flags",
    ]
    table = pq.read_table(root / "reference_interactions.parquet", columns=columns, use_threads=True)
    try:
        n = table.num_rows
        if n != int(manifest["sample"]["reference_interaction_rows"]):
            raise CalibrationError("Loaded interaction count differs from manifest")
        learner = arrow_to_numpy_int(table["learner_id"], np.int32)
        skill = arrow_to_numpy_int(table["skill_id"], np.int32)
        within = arrow_to_numpy_int(table["within_skill_index"], np.int32)
        gap = arrow_to_numpy_int(table["gap_ms"], np.int64)
        quality = arrow_to_numpy_int(table["quality_flags"], np.int32)
        correctness = table["is_correct"].combine_chunks()
        if correctness.null_count:
            raise CalibrationError("Null correctness in reference interactions")
        y = np.asarray(correctness.to_numpy(zero_copy_only=False), dtype=np.uint8)
        split = np.empty(n, dtype=np.int8)
        for code, name in enumerate(SPLIT_NAMES):
            mask = np.asarray(pc.equal(table["split"], pa.scalar(name)).to_numpy(zero_copy_only=False), dtype=bool)
            split[mask] = code
        recognized_split = np.zeros(n, dtype=bool)
        for name in SPLIT_NAMES:
            recognized_split |= np.asarray(pc.equal(table["split"], pa.scalar(name)).to_numpy(zero_copy_only=False), dtype=bool)
        if not np.all(recognized_split):
            raise CalibrationError("Unknown split label in reference interactions")
        sample = np.empty(n, dtype=np.int8)
        recognized_class = np.zeros(n, dtype=bool)
        for code, name in enumerate(CLASS_NAMES):
            mask = np.asarray(pc.equal(table["sample_class"], pa.scalar(name)).to_numpy(zero_copy_only=False), dtype=bool)
            sample[mask] = code
            recognized_class |= mask
        if not np.all(recognized_class):
            raise CalibrationError("Unknown sample class in reference interactions")
        excluded_mask = int(manifest["filter"]["excluded_quality_mask"])
        if np.any((quality & excluded_mask) != 0):
            raise CalibrationError("Reference interactions contain excluded quality bits")
        if np.any(gap < 0):
            raise CalibrationError("Negative gap in reference interactions")
        if np.any((y != 0) & (y != 1)):
            raise CalibrationError("Non-binary correctness in reference interactions")
        selected_skills = np.asarray(manifest["selection"]["selected_skills"], dtype=np.int32)
        if not np.all(np.isin(skill, selected_skills)):
            raise CalibrationError("Reference interactions contain unselected skills")
        # Validate question membership without materializing millions of strings in Python.
        selected_item_rows = read_csv_rows(root / "selected_items.csv")
        selected_q = pa.array([r["question_id"] for r in selected_item_rows])
        member = pc.is_in(table["question_id"], value_set=selected_q)
        if not bool(pc.all(member).as_py()):
            raise CalibrationError("Reference interactions contain a question outside selected_items.csv")
        return {
            "learner_id": learner,
            "skill_id": skill,
            "split_code": split,
            "sample_code": sample,
            "within_skill_index": within,
            "gap_ms": gap,
            "outcome": y,
            "quality_flags": quality,
        }
    finally:
        del table


def validate_split_and_pair_consistency(root: Path, arrays: dict[str, np.ndarray], manifest: dict[str, Any]) -> None:
    pa, pc, pq = require_pyarrow()
    learner = arrays["learner_id"]
    skill = arrays["skill_id"]
    split = arrays["split_code"]
    sample = arrays["sample_code"]
    within = arrays["within_skill_index"]

    # Learner split map.
    split_table = pq.read_table(root / "learner_splits.parquet", columns=["learner_id", "split"])
    try:
        ids = arrow_to_numpy_int(split_table["learner_id"], np.int32)
        max_id = int(max(ids.max(initial=0), learner.max(initial=0)))
        split_map = np.full(max_id + 1, -1, dtype=np.int8)
        for code, name in enumerate(SPLIT_NAMES):
            mask = np.asarray(pc.equal(split_table["split"], pa.scalar(name)).to_numpy(zero_copy_only=False), dtype=bool)
            split_map[ids[mask]] = code
        if np.any(split_map[learner] != split):
            raise CalibrationError("Interaction split differs from learner_splits.parquet")
    finally:
        del split_table

    max_skill = int(max(skill.max(initial=0), max(manifest["selection"]["selected_skills"]))) + 1
    key = learner.astype(np.int64) * max_skill + skill.astype(np.int64)
    order = np.lexsort((within, key))
    key_s = key[order]
    within_s = within[order]
    split_s = split[order]
    sample_s = sample[order]
    starts = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1]])
    ends = np.r_[starts[1:], key_s.size]
    counts = ends - starts
    if starts.size != int(manifest["sample"]["selected_learner_skill_pairs"]):
        raise CalibrationError("Unique learner-skill pair count differs from manifest")
    for st, en, count in zip(starts, ends, counts):
        idx = within_s[st:en]
        if not np.array_equal(idx, np.arange(count, dtype=idx.dtype)):
            raise CalibrationError("within_skill_index is not contiguous for a learner-skill pair")
        if np.any(split_s[st:en] != split_s[st]) or np.any(sample_s[st:en] != sample_s[st]):
            raise CalibrationError("Split or sample class changes within a learner-skill sequence")
        expected_class = 0 if count <= 2 else 1
        if int(sample_s[st]) != expected_class:
            raise CalibrationError("sample_class is inconsistent with sequence length")

    pair_table = pq.read_table(root / "selected_learner_skill_pairs.parquet")
    try:
        pl = arrow_to_numpy_int(pair_table["learner_id"], np.int32)
        ps = arrow_to_numpy_int(pair_table["skill_id"], np.int32)
        pe = arrow_to_numpy_int(pair_table["event_count"], np.int32)
        pkey = pl.astype(np.int64) * max_skill + ps.astype(np.int64)
        pair_order = np.argsort(pkey, kind="stable")
        unique_key = key_s[starts]
        count_order = np.argsort(unique_key, kind="stable")
        if not np.array_equal(pkey[pair_order], unique_key[count_order]):
            raise CalibrationError("Selected pair keys differ from extracted interactions")
        if not np.array_equal(pe[pair_order].astype(np.int64), counts[count_order].astype(np.int64)):
            raise CalibrationError("Selected pair event counts differ from extracted interactions")
    finally:
        del pair_table


def build_batch(arrays: dict[str, np.ndarray], skill_id: int, split_code: int, sample_code: int | None = None) -> SequenceBatch:
    mask = (arrays["skill_id"] == skill_id) & (arrays["split_code"] == split_code)
    if sample_code is not None:
        mask &= arrays["sample_code"] == sample_code
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise CalibrationError(f"No rows for skill={skill_id}, split={SPLIT_NAMES[split_code]}")
    learner = arrays["learner_id"][idx]
    within = arrays["within_skill_index"][idx]
    order = np.lexsort((within, learner))
    idx = idx[order]
    learner = learner[order]
    within = within[order]
    starts_orig = np.flatnonzero(np.r_[True, learner[1:] != learner[:-1]])
    ends_orig = np.r_[starts_orig[1:], learner.size]
    lengths_orig = (ends_orig - starts_orig).astype(np.int32)
    # Deterministic descending-length order reduces vectorized recursion work.
    sequence_learners = learner[starts_orig]
    sequence_samples = arrays["sample_code"][idx[starts_orig]]
    seq_order = np.lexsort((sequence_learners, -lengths_orig))
    lengths = lengths_orig[seq_order]
    ordered_learners = sequence_learners[seq_order]
    ordered_samples = sequence_samples[seq_order]
    pieces = [idx[starts_orig[j]:ends_orig[j]] for j in seq_order]
    flat_idx = np.concatenate(pieces)
    starts = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(lengths[:-1], dtype=np.int64)))
    gap_ms = arrays["gap_ms"][flat_idx].astype(np.int64, copy=True)
    outcomes = arrays["outcome"][flat_idx].astype(np.uint8, copy=True)
    # The reference builder defines first gap as zero; enforce validation, not silent repair.
    batch = SequenceBatch(
        lengths=lengths,
        starts=starts,
        gaps_days=gap_ms.astype(np.float64) / DAY_MS,
        gaps_ms=gap_ms,
        outcomes=outcomes,
        learner_ids=ordered_learners.astype(np.int32, copy=True),
        sample_codes=ordered_samples.astype(np.int8, copy=True),
        skill_id=int(skill_id),
        split_name=SPLIT_NAMES[split_code],
    )
    batch.validate()
    return batch


def sigmoid_bounded(z: float, upper: float = 1.0) -> tuple[float, float]:
    x = float(expit(z))
    value = upper * x
    derivative = upper * x * (1.0 - x)
    return value, derivative


def unpack_raw(z: np.ndarray, forgetting: bool) -> tuple[np.ndarray, np.ndarray]:
    p0, dp0 = sigmoid_bounded(float(z[0]))
    pL, dpL = sigmoid_bounded(float(z[1]))
    s, ds = sigmoid_bounded(float(z[2]), MAX_EMISSION_ERROR)
    g, dg = sigmoid_bounded(float(z[3]), MAX_EMISSION_ERROR)
    if forgetting:
        lam = float(math.exp(float(z[4])))
        return np.array([p0, pL, s, g, lam]), np.array([dp0, dpL, ds, dg, lam])
    return np.array([p0, pL, s, g]), np.array([dp0, dpL, ds, dg])


def natural_to_raw(params: Sequence[float], forgetting: bool, config: FitConfig) -> np.ndarray:
    p0, pL, s, g = [float(x) for x in params[:4]]
    vals = [
        logit(np.clip(p0, 1e-8, 1 - 1e-8)),
        logit(np.clip(pL, 1e-8, 1 - 1e-8)),
        logit(np.clip(s / MAX_EMISSION_ERROR, 1e-8, 1 - 1e-8)),
        logit(np.clip(g / MAX_EMISSION_ERROR, 1e-8, 1 - 1e-8)),
    ]
    if forgetting:
        vals.append(math.log(np.clip(float(params[4]), config.lambda_min, config.lambda_max)))
    return np.asarray(vals, dtype=np.float64)


def nll_gradient_natural(params: np.ndarray, batch: SequenceBatch, forgetting: bool) -> tuple[float, np.ndarray]:
    p0, pL, slip, guess = [float(x) for x in params[:4]]
    lam = float(params[4]) if forgetting else 0.0
    k = 5 if forgetting else 4
    nseq = batch.n_sequences
    state = np.full(nseq, p0, dtype=np.float64)
    derivative = np.zeros((nseq, k), dtype=np.float64)
    derivative[:, 0] = 1.0
    grad = np.zeros(k, dtype=np.float64)
    nll = 0.0
    lengths = batch.lengths
    starts = batch.starts
    outcomes = batch.outcomes
    gaps = batch.gaps_days
    for t in range(batch.max_length):
        n_active = int(np.searchsorted(-lengths, -t, side="left"))
        if n_active <= 0:
            break
        positions = starts[:n_active] + t
        d = gaps[positions]
        y = outcomes[positions].astype(np.float64, copy=False)
        if forgetting:
            retention = np.exp(-lam * d)
        else:
            retention = np.ones_like(d)
        pre = state[:n_active] * retention
        dpre = derivative[:n_active] * retention[:, None]
        if forgetting:
            dpre[:, 4] += state[:n_active] * (-d * retention)
        q = guess + pre * (1.0 - slip - guess)
        q = np.clip(q, EPS, 1.0 - EPS)
        dq = dpre * (1.0 - slip - guess)
        dq[:, 2] -= pre
        dq[:, 3] += 1.0 - pre
        nll -= float(np.sum(y * np.log(q) + (1.0 - y) * np.log1p(-q)))
        score_q = (q - y) / (q * (1.0 - q))
        grad += np.sum(score_q[:, None] * dq, axis=0)

        l_mastered = y * (1.0 - slip) + (1.0 - y) * slip
        l_unmastered = y * guess + (1.0 - y) * (1.0 - guess)
        denominator = pre * l_mastered + (1.0 - pre) * l_unmastered
        denominator = np.clip(denominator, EPS, None)
        numerator = pre * l_mastered
        posterior = numerator / denominator
        dl_mastered = np.zeros((n_active, k), dtype=np.float64)
        dl_unmastered = np.zeros((n_active, k), dtype=np.float64)
        dl_mastered[:, 2] = 1.0 - 2.0 * y
        dl_unmastered[:, 3] = 2.0 * y - 1.0
        dnumerator = dpre * l_mastered[:, None] + pre[:, None] * dl_mastered
        ddenominator = (
            dpre * (l_mastered - l_unmastered)[:, None]
            + pre[:, None] * dl_mastered
            + (1.0 - pre)[:, None] * dl_unmastered
        )
        dposterior = (
            dnumerator * denominator[:, None] - numerator[:, None] * ddenominator
        ) / (denominator[:, None] ** 2)
        state_new = pL + (1.0 - pL) * posterior
        derivative_new = (1.0 - pL) * dposterior
        derivative_new[:, 1] += 1.0 - posterior
        state[:n_active] = state_new
        derivative[:n_active] = derivative_new
    if not np.isfinite(nll) or np.any(~np.isfinite(grad)):
        raise FloatingPointError("Non-finite likelihood or gradient")
    return nll, grad


def objective_raw(z: np.ndarray, batch: SequenceBatch, forgetting: bool) -> tuple[float, np.ndarray]:
    natural, chain = unpack_raw(z, forgetting)
    nll, grad_natural = nll_gradient_natural(natural, batch, forgetting)
    return nll, grad_natural * chain


def predict_probabilities(params: np.ndarray, batch: SequenceBatch, forgetting: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p0, pL, slip, guess = [float(x) for x in params[:4]]
    lam = float(params[4]) if forgetting else 0.0
    state = np.full(batch.n_sequences, p0, dtype=np.float64)
    probs = np.empty(batch.n_events, dtype=np.float64)
    event_gaps = np.empty(batch.n_events, dtype=np.int64)
    event_y = np.empty(batch.n_events, dtype=np.uint8)
    out_pos = 0
    for t in range(batch.max_length):
        n_active = int(np.searchsorted(-batch.lengths, -t, side="left"))
        if n_active <= 0:
            break
        positions = batch.starts[:n_active] + t
        d = batch.gaps_days[positions]
        y = batch.outcomes[positions].astype(np.float64, copy=False)
        retention = np.exp(-lam * d) if forgetting else np.ones_like(d)
        pre = state[:n_active] * retention
        q = np.clip(guess + pre * (1.0 - slip - guess), EPS, 1.0 - EPS)
        probs[out_pos:out_pos+n_active] = q
        event_gaps[out_pos:out_pos+n_active] = batch.gaps_ms[positions]
        event_y[out_pos:out_pos+n_active] = batch.outcomes[positions]
        out_pos += n_active
        l_mastered = y * (1.0 - slip) + (1.0 - y) * slip
        l_unmastered = y * guess + (1.0 - y) * (1.0 - guess)
        den = np.clip(pre * l_mastered + (1.0 - pre) * l_unmastered, EPS, None)
        posterior = pre * l_mastered / den
        state[:n_active] = pL + (1.0 - pL) * posterior
    if out_pos != batch.n_events:
        raise CalibrationError("Prediction traversal did not cover every event")
    return probs, event_y, event_gaps


def auc_binary(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.uint8)
    n1 = int(y.sum())
    n0 = int(y.size - n1)
    if n1 == 0 or n0 == 0:
        return math.nan
    ranks = rankdata(p, method="average")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float, str]:
    if np.unique(y).size < 2:
        return math.nan, math.nan, "single_outcome"
    x = np.clip(np.log(p / (1.0 - p)), -20.0, 20.0)
    X = np.column_stack((np.ones_like(x), x))
    beta = np.array([0.0, 1.0], dtype=np.float64)
    status = "maxiter"
    for _ in range(80):
        eta = X @ beta
        mu = expit(eta)
        w = np.clip(mu * (1.0 - mu), 1e-8, None)
        grad = X.T @ (mu - y)
        hess = X.T @ (w[:, None] * X)
        hess += np.eye(2) * 1e-8
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return math.nan, math.nan, "singular"
        beta_new = beta - step
        if np.any(~np.isfinite(beta_new)):
            return math.nan, math.nan, "non_finite"
        if np.max(np.abs(step)) < 1e-9:
            beta = beta_new
            status = "converged"
            break
        beta = beta_new
    return float(beta[0]), float(beta[1]), status


def prediction_metrics(y: np.ndarray, p: np.ndarray, ece_bins: int) -> dict[str, Any]:
    y_f = y.astype(np.float64)
    p = np.clip(p.astype(np.float64), EPS, 1.0 - EPS)
    logloss = -float(np.mean(y_f * np.log(p) + (1.0 - y_f) * np.log1p(-p)))
    brier = float(np.mean((p - y_f) ** 2))
    accuracy = float(np.mean((p >= 0.5) == y))
    ece = 0.0
    bins = np.linspace(0.0, 1.0, ece_bins + 1)
    ids = np.minimum(np.searchsorted(bins, p, side="right") - 1, ece_bins - 1)
    for b in range(ece_bins):
        mask = ids == b
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(p[mask].mean()) - float(y_f[mask].mean()))
    intercept, slope, cal_status = calibration_intercept_slope(y_f, p)
    return {
        "n_events": int(y.size),
        "event_rate": float(y_f.mean()),
        "mean_prediction": float(p.mean()),
        "log_loss": logloss,
        "brier": brier,
        "accuracy_0_5": accuracy,
        "auc": auc_binary(y, p),
        "ece_equal_width": float(ece),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "calibration_fit_status": cal_status,
    }


def calibration_bin_rows(skill_id: int, model: str, split: str, y: np.ndarray, p: np.ndarray, n_bins: int) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.minimum(np.searchsorted(edges, p, side="right") - 1, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = ids == b
        rows.append({
            "skill_id": skill_id,
            "model": model,
            "split": split,
            "bin_index": b,
            "bin_lower": float(edges[b]),
            "bin_upper": float(edges[b + 1]),
            "n_events": int(mask.sum()),
            "mean_prediction": float(p[mask].mean()) if np.any(mask) else math.nan,
            "observed_rate": float(y[mask].mean()) if np.any(mask) else math.nan,
        })
    return rows


def gap_metric_rows(skill_id: int, model: str, split: str, y: np.ndarray, p: np.ndarray, gaps_ms: np.ndarray, ece_bins: int) -> list[dict[str, Any]]:
    rows = []
    for label, lower, upper in GAP_BINS:
        if label == "first_or_zero":
            mask = gaps_ms <= 0
        else:
            mask = (gaps_ms > lower) & ((gaps_ms <= upper) if math.isfinite(upper) else True)
        if not np.any(mask):
            metrics = {"n_events": 0, "event_rate": math.nan, "mean_prediction": math.nan, "log_loss": math.nan,
                       "brier": math.nan, "accuracy_0_5": math.nan, "auc": math.nan, "ece_equal_width": math.nan,
                       "calibration_intercept": math.nan, "calibration_slope": math.nan, "calibration_fit_status": "empty"}
        else:
            metrics = prediction_metrics(y[mask], p[mask], ece_bins)
        rows.append({"skill_id": skill_id, "model": model, "split": split, "gap_bin": label, **metrics})
    return rows


def deterministic_starts(batch: SequenceBatch, forgetting: bool, count: int, config: FitConfig) -> list[np.ndarray]:
    # First-response accuracy supplies a data-informed but non-binding start.
    first_y = batch.outcomes[batch.starts]
    first_rate = float(first_y.mean())
    templates = [
        (0.10, 0.05, 0.05, 0.15, 0.001),
        (0.20, 0.10, 0.10, 0.20, 0.005),
        (0.35, 0.15, 0.10, 0.25, 0.010),
        (0.50, 0.20, 0.15, 0.20, 0.030),
        (0.25, 0.30, 0.05, 0.30, 0.080),
        (0.60, 0.08, 0.20, 0.15, 0.150),
        (0.40, 0.35, 0.08, 0.10, 0.300),
        (0.15, 0.20, 0.20, 0.30, 0.600),
        (0.70, 0.05, 0.05, 0.25, 1.000),
        (0.30, 0.45, 0.15, 0.10, 0.020),
        (0.55, 0.25, 0.25, 0.20, 0.100),
        (0.05, 0.12, 0.08, 0.35, 0.400),
    ]
    # Add a start adjusted to first-response accuracy under moderate emission error.
    s0, g0 = 0.10, 0.20
    p0_est = float(np.clip((first_rate - g0) / max(1.0 - s0 - g0, 1e-6), 0.03, 0.90))
    templates[0] = (p0_est, 0.12, s0, g0, 0.03)
    result = []
    for tpl in templates[:count]:
        if forgetting:
            result.append(natural_to_raw(tpl, True, config))
        else:
            result.append(natural_to_raw(tpl[:4], False, config))
    return result


def raw_bounds(forgetting: bool, config: FitConfig) -> list[tuple[float, float]]:
    b = [(-9.0, 9.0), (-9.0, 9.0), (-9.0, 9.0), (-9.0, 9.0)]
    if forgetting:
        b.append((math.log(config.lambda_min), math.log(config.lambda_max)))
    return b


def fit_model(batch: SequenceBatch, forgetting: bool, config: FitConfig) -> dict[str, Any]:
    starts = deterministic_starts(batch, forgetting, config.starts_bktf if forgetting else config.starts_bkt, config)
    start_rows: list[dict[str, Any]] = []
    best = None
    for i, z0 in enumerate(starts):
        t0 = time.perf_counter()
        result = minimize(
            fun=lambda z: objective_raw(z, batch, forgetting),
            x0=z0,
            method="L-BFGS-B",
            jac=True,
            bounds=raw_bounds(forgetting, config),
            options={"maxiter": config.maxiter, "ftol": config.ftol, "gtol": config.gtol, "maxls": 40},
        )
        elapsed = time.perf_counter() - t0
        natural, _ = unpack_raw(result.x, forgetting)
        row = {
            "start_index": i,
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nll": float(result.fun),
            "nit": int(result.nit),
            "nfev": int(result.nfev),
            "njev": int(getattr(result, "njev", -1)),
            "elapsed_seconds": elapsed,
            "parameters": ModelParameters(
                float(natural[0]), float(natural[1]), float(natural[2]), float(natural[3]),
                float(natural[4]) if forgetting else 0.0,
            ).as_dict(),
            "raw_parameters": [float(x) for x in result.x],
            "gradient_inf_norm": float(np.max(np.abs(result.jac))),
        }
        start_rows.append(row)
        if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        raise CalibrationError("Every optimization start failed with non-finite objective")
    natural, chain = unpack_raw(best.x, forgetting)
    k = natural.size
    # Numerical Hessian of the analytic raw gradient for approximate local uncertainty.
    hessian_status = "not_computed"
    se_natural = np.full(k, np.nan)
    eigenvalues = np.full(k, np.nan)
    try:
        h = np.empty((k, k), dtype=np.float64)
        step_base = 1e-4
        for j in range(k):
            step = step_base * max(1.0, abs(float(best.x[j])))
            zp = best.x.copy(); zm = best.x.copy()
            zp[j] += step; zm[j] -= step
            gp = objective_raw(zp, batch, forgetting)[1]
            gm = objective_raw(zm, batch, forgetting)[1]
            h[:, j] = (gp - gm) / (2.0 * step)
        h = 0.5 * (h + h.T)
        eigenvalues = np.linalg.eigvalsh(h)
        if np.all(eigenvalues > 1e-8):
            cov_raw = np.linalg.inv(h)
            cov_nat = np.diag(chain) @ cov_raw @ np.diag(chain)
            se_natural = np.sqrt(np.clip(np.diag(cov_nat), 0.0, None))
            hessian_status = "positive_definite"
        else:
            hessian_status = "not_positive_definite"
    except Exception as exc:
        hessian_status = f"failed:{type(exc).__name__}"
    params = ModelParameters(
        float(natural[0]), float(natural[1]), float(natural[2]), float(natural[3]),
        float(natural[4]) if forgetting else 0.0,
    )
    return {
        "model": "BKT-F" if forgetting else "BKT",
        "forgetting": forgetting,
        "optimizer_success": bool(best.success),
        "optimizer_status": int(best.status),
        "optimizer_message": str(best.message),
        "nll_train": float(best.fun),
        "n_parameters": int(k),
        "n_iterations": int(best.nit),
        "n_function_evaluations": int(best.nfev),
        "gradient_inf_norm": float(np.max(np.abs(best.jac))),
        "parameters": params.as_dict(),
        "parameter_se_approx": {
            "p_init": float(se_natural[0]), "p_learn": float(se_natural[1]),
            "slip": float(se_natural[2]), "guess": float(se_natural[3]),
            "lambda_per_day": float(se_natural[4]) if forgetting else math.nan,
        },
        "hessian_status": hessian_status,
        "hessian_eigenvalues": [float(x) for x in eigenvalues],
        "raw_parameters": [float(x) for x in best.x],
        "all_starts": start_rows,
    }


def model_natural_array(fit: dict[str, Any]) -> np.ndarray:
    p = fit["parameters"]
    if fit["forgetting"]:
        return np.array([p["p_init"], p["p_learn"], p["slip"], p["guess"], p["lambda_per_day"]], dtype=np.float64)
    return np.array([p["p_init"], p["p_learn"], p["slip"], p["guess"]], dtype=np.float64)


def evaluate_fit(skill_id: int, fit: dict[str, Any], batches: dict[tuple[int, int | None], SequenceBatch], config: FitConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    params = model_natural_array(fit)
    forgetting = bool(fit["forgetting"])
    model = fit["model"]
    metrics_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    cal_rows: list[dict[str, Any]] = []
    for split_code, split_name in enumerate(SPLIT_NAMES):
        for sample_code in (None, 0, 1):
            batch = batches[(split_code, sample_code)]
            probs, y, gaps = predict_probabilities(params, batch, forgetting)
            metrics = prediction_metrics(y, probs, config.ece_bins)
            sample_name = "all" if sample_code is None else CLASS_NAMES[sample_code]
            metrics_rows.append({
                "skill_id": skill_id, "model": model, "split": split_name,
                "sample_class": sample_name, "n_sequences": batch.n_sequences, **metrics,
            })
            if sample_code is None:
                gap_rows.extend(gap_metric_rows(skill_id, model, split_name, y, probs, gaps, config.ece_bins))
                cal_rows.extend(calibration_bin_rows(skill_id, model, split_name, y, probs, config.ece_bins))
    return metrics_rows, gap_rows, cal_rows


def skill_result_path(work: Path, skill_id: int) -> Path:
    return work / "skills" / f"skill_{skill_id:03d}.json"


def fit_one_skill(skill_id: int, arrays: dict[str, np.ndarray], config: FitConfig, identity_hash: str, work: Path) -> dict[str, Any]:
    result_path = skill_result_path(work, skill_id)
    if result_path.is_file():
        try:
            saved = read_json(result_path)
        except Exception:
            corrupt = result_path.with_suffix(result_path.suffix + ".corrupt")
            result_path.replace(corrupt)
            saved = None
        if isinstance(saved, dict) and saved.get("identity_hash") == identity_hash and saved.get("status") == "PASS":
            print(f"SKILL {skill_id}: resume PASS", flush=True)
            return saved
        result_path.unlink(missing_ok=True)
    print(f"SKILL {skill_id}: building sequence batches", flush=True)
    batches: dict[tuple[int, int | None], SequenceBatch] = {}
    for split_code in range(3):
        batches[(split_code, None)] = build_batch(arrays, skill_id, split_code, None)
        batches[(split_code, 0)] = build_batch(arrays, skill_id, split_code, 0)
        batches[(split_code, 1)] = build_batch(arrays, skill_id, split_code, 1)
    train = batches[(0, None)]
    print(f"SKILL {skill_id}: BKT fit | sequences={train.n_sequences:,} events={train.n_events:,}", flush=True)
    bkt = fit_model(train, False, config)
    print(f"SKILL {skill_id}: BKT-F fit", flush=True)
    bktf = fit_model(train, True, config)
    metrics_bkt, gaps_bkt, bins_bkt = evaluate_fit(skill_id, bkt, batches, config)
    metrics_bktf, gaps_bktf, bins_bktf = evaluate_fit(skill_id, bktf, batches, config)
    cal_bkt = next(r for r in metrics_bkt if r["split"] == "calibration" and r["sample_class"] == "all")
    cal_bktf = next(r for r in metrics_bktf if r["split"] == "calibration" and r["sample_class"] == "all")
    gain = float(cal_bkt["log_loss"] - cal_bktf["log_loss"])
    lam = float(bktf["parameters"]["lambda_per_day"])
    preferred = "BKT-F" if cal_bktf["log_loss"] < cal_bkt["log_loss"] else "BKT"
    conservative_support = bool(
        gain >= config.min_calibration_logloss_gain
        and lam > config.lambda_min * 10.0
        and bktf["optimizer_success"]
    )
    result = {
        "status": "PASS",
        "identity_hash": identity_hash,
        "skill_id": skill_id,
        "created_at_utc": utc_iso(),
        "batch_summary": {
            f"{SPLIT_NAMES[s]}_{'all' if c is None else CLASS_NAMES[c]}": {
                "n_sequences": batches[(s, c)].n_sequences,
                "n_events": batches[(s, c)].n_events,
                "max_length": batches[(s, c)].max_length,
            }
            for s in range(3) for c in (None, 0, 1)
        },
        "fits": {"BKT": bkt, "BKT-F": bktf},
        "metrics": metrics_bkt + metrics_bktf,
        "gap_metrics": gaps_bkt + gaps_bktf,
        "calibration_bins": bins_bkt + bins_bktf,
        "model_selection": {
            "preferred_by_calibration_log_loss": preferred,
            "calibration_log_loss_gain_bktf_over_bkt": gain,
            "minimum_gain_for_conservative_support": config.min_calibration_logloss_gain,
            "forgetting_supported_conservative": conservative_support,
        },
    }
    atomic_json(result_path, result)
    print(f"SKILL {skill_id}: PASS | preferred={preferred} | cal logloss gain={gain:.8f}", flush=True)
    return result


def flatten_results(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fit_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for result in results:
        skill = int(result["skill_id"])
        for model_name, fit in result["fits"].items():
            p = fit["parameters"]
            se = fit["parameter_se_approx"]
            n_train = result["batch_summary"]["train_all"]["n_events"]
            k = fit["n_parameters"]
            fit_rows.append({
                "skill_id": skill, "model": model_name,
                "optimizer_success": fit["optimizer_success"], "optimizer_status": fit["optimizer_status"],
                "optimizer_message": fit["optimizer_message"], "n_train_events": n_train,
                "n_train_sequences": result["batch_summary"]["train_all"]["n_sequences"],
                "nll_train": fit["nll_train"], "log_loss_train_objective": fit["nll_train"] / n_train,
                "n_parameters": k, "aic_train": 2 * k + 2 * fit["nll_train"],
                "bic_train": k * math.log(n_train) + 2 * fit["nll_train"],
                "n_iterations": fit["n_iterations"], "n_function_evaluations": fit["n_function_evaluations"],
                "gradient_inf_norm": fit["gradient_inf_norm"], "hessian_status": fit["hessian_status"],
                "p_init": p["p_init"], "p_learn": p["p_learn"], "slip": p["slip"], "guess": p["guess"],
                "lambda_per_day": p["lambda_per_day"], "retention_half_life_days": p["retention_half_life_days"],
                "se_p_init_approx": se["p_init"], "se_p_learn_approx": se["p_learn"],
                "se_slip_approx": se["slip"], "se_guess_approx": se["guess"],
                "se_lambda_per_day_approx": se["lambda_per_day"],
            })
        metric_rows.extend(result["metrics"])
        gap_rows.extend(result["gap_metrics"])
        bin_rows.extend(result["calibration_bins"])
        selection_rows.append({"skill_id": skill, **result["model_selection"]})
    return fit_rows, metric_rows, gap_rows, bin_rows, selection_rows


def aggregate_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    keys = sorted({(r["model"], r["split"], r["sample_class"]) for r in metric_rows})
    metric_names = ["log_loss", "brier", "accuracy_0_5", "auc", "ece_equal_width", "calibration_intercept", "calibration_slope"]
    for model, split, sample_class in keys:
        rows = [r for r in metric_rows if r["model"] == model and r["split"] == split and r["sample_class"] == sample_class]
        total = sum(int(r["n_events"]) for r in rows)
        base = {"model": model, "split": split, "sample_class": sample_class, "n_skills": len(rows), "n_events": total}
        for m in metric_names:
            vals = np.array([float(r[m]) if r.get(m) is not None else math.nan for r in rows], dtype=float)
            weights = np.array([int(r["n_events"]) for r in rows], dtype=float)
            valid = np.isfinite(vals)
            base[f"macro_{m}"] = float(np.mean(vals[valid])) if np.any(valid) else math.nan
            base[f"micro_weighted_{m}"] = float(np.average(vals[valid], weights=weights[valid])) if np.any(valid) else math.nan
        output.append(base)
    return output


def build_output_bundle(final_dir: Path, input_identity: str, config: FitConfig, manifest: dict[str, Any], results: list[dict[str, Any]], elapsed: float) -> Path:
    fit_rows, metric_rows, gap_rows, bin_rows, selection_rows = flatten_results(results)
    aggregate = aggregate_rows(metric_rows)
    preferred_params = []
    for result in results:
        preferred = result["model_selection"]["preferred_by_calibration_log_loss"]
        fit = result["fits"][preferred]
        preferred_params.append({"skill_id": result["skill_id"], "selected_model": preferred, **fit["parameters"],
                                 "forgetting_supported_conservative": result["model_selection"]["forgetting_supported_conservative"]})
    final_dir.mkdir(parents=True, exist_ok=True)
    csv_specs = [
        ("skill_model_fits.csv", fit_rows), ("split_metrics.csv", metric_rows),
        ("gap_metrics.csv", gap_rows), ("calibration_bins.csv", bin_rows),
        ("model_selection.csv", selection_rows), ("aggregate_metrics.csv", aggregate),
        ("preferred_world_parameters.csv", preferred_params),
    ]
    for name, rows in csv_specs:
        fields = list(rows[0].keys()) if rows else []
        atomic_csv(final_dir / name, fields, rows)
    models_dir = final_dir / "models"
    models_dir.mkdir(exist_ok=True)
    for result in results:
        atomic_json(models_dir / f"skill_{int(result['skill_id']):03d}.json", result)
    output_manifest = {
        "created_at_utc": utc_iso(),
        "input_identity_hash": input_identity,
        "input_reference_manifest_canonical_sha256": hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "pyarrow": __import__("pyarrow").__version__,
        },
        "config": config.as_dict(),
        "selected_skills": [int(x) for x in manifest["selection"]["selected_skills"]],
        "skills_completed": len(results),
        "elapsed_seconds": elapsed,
        "model_selection_rule": {
            "primary": "lower learner-disjoint calibration log loss",
            "conservative_forgetting_support": f"BKT-F gain >= {config.min_calibration_logloss_gain} per event and lambda above numerical boundary",
            "test_split_role": "locked external evaluation; not used for model selection",
        },
        "uncertainty_note": "Reported standard errors are local observed-Hessian approximations. Learner-cluster bootstrap is required before inferential manuscript claims.",
        "status": "PASS",
    }
    atomic_json(final_dir / "calibration_manifest.json", output_manifest)
    readme = f"""# AdaptiveLearningSim BKT/BKT-F calibration bundle\n\nStatus: PASS\n\n- Skills: {len(results)}\n- Input reference interactions: {manifest['sample']['reference_interaction_rows']:,}\n- Models per skill: BKT and BKT-F\n- Selection: learner-disjoint calibration log loss\n- Test split was not used for fitting or model selection.\n\nThe Hessian-based standard errors are diagnostic approximations, not final\ninferential uncertainty estimates. A learner-cluster bootstrap is required\nbefore manuscript-level confidence intervals are reported.\n"""
    (final_dir / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    hash_lines = []
    for p in sorted(final_dir.rglob("*")):
        if p.is_file() and p.name not in {"SHA256SUMS.txt", "AdaptiveLearningSim_BKTF_calibration_bundle.zip"}:
            hash_lines.append(f"{sha256_file(p)}  {p.relative_to(final_dir).as_posix()}")
    (final_dir / "SHA256SUMS.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8", newline="\n")
    zip_path = final_dir / "AdaptiveLearningSim_BKTF_calibration_bundle.zip"
    tmp_zip = zip_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(final_dir.rglob("*")):
            if p.is_file() and p not in {zip_path, tmp_zip}:
                zf.write(p, p.relative_to(final_dir).as_posix())
    os.replace(tmp_zip, zip_path)
    return zip_path


def simulate_batch(params: ModelParameters, n_sequences: int, length: int, seed: int) -> SequenceBatch:
    rng = np.random.default_rng(seed)
    lengths = np.full(n_sequences, length, dtype=np.int32)
    starts = np.arange(n_sequences, dtype=np.int64) * length
    gaps = np.zeros(n_sequences * length, dtype=np.float64)
    y = np.zeros(n_sequences * length, dtype=np.uint8)
    for i in range(n_sequences):
        mastered = bool(rng.random() < params.p_init)
        for t in range(length):
            pos = i * length + t
            if t > 0:
                # Mixture of within-session and multi-day gaps supports lambda identification.
                if rng.random() < 0.55:
                    d = rng.uniform(1.0 / 1440.0, 0.05)
                else:
                    d = rng.uniform(0.5, 12.0)
                gaps[pos] = d
                if mastered and rng.random() < 1.0 - math.exp(-params.lambda_per_day * d):
                    mastered = False
            prob = (1.0 - params.slip) if mastered else params.guess
            yy = rng.random() < prob
            y[pos] = int(yy)
            if (not mastered) and rng.random() < params.p_learn:
                mastered = True
    batch = SequenceBatch(
        lengths=lengths, starts=starts, gaps_days=gaps,
        gaps_ms=np.rint(gaps * DAY_MS).astype(np.int64), outcomes=y,
        learner_ids=np.arange(n_sequences, dtype=np.int32), sample_codes=np.ones(n_sequences, dtype=np.int8),
        skill_id=1, split_name="synthetic",
    )
    batch.validate()
    return batch


def run_self_test() -> None:
    from scipy.optimize._numdiff import approx_derivative
    # Real Parquet write/read/close smoke test. This catches binary dependency and
    # Windows file-handle issues before the reference bundle is opened.
    pa, _, pq = require_pyarrow()
    parquet_temp = Path(tempfile.mkdtemp(prefix="als_bktf_parquet_selftest_"))
    try:
        ptest = parquet_temp / "smoke.parquet"
        pq.write_table(pa.table({"a": pa.array([1, 2, 3], type=pa.int32()), "b": [True, False, True]}), ptest, compression="zstd")
        rows, cols, groups = parquet_metadata_and_schema(ptest)
        if rows != 3 or cols != ["a", "b"] or groups != 1:
            raise CalibrationError("Parquet smoke-test metadata mismatch")
        table = pq.read_table(ptest)
        try:
            if table.num_rows != 3 or table.column_names != ["a", "b"]:
                raise CalibrationError("Parquet smoke-test read mismatch")
        finally:
            del table
    finally:
        cleanup_tree(parquet_temp)
    true = ModelParameters(0.25, 0.18, 0.09, 0.21, 0.06)
    batch = simulate_batch(true, n_sequences=1200, length=24, seed=771)
    config = FitConfig(starts_bkt=4, starts_bktf=6, maxiter=250)
    z = natural_to_raw(true.as_array(True), True, config)
    value, grad = objective_raw(z, batch, True)
    numeric = approx_derivative(lambda x: np.array([objective_raw(x, batch, True)[0]]), z, method="3-point").ravel()
    rel = np.max(np.abs(grad - numeric) / (1.0 + np.abs(numeric)))
    if not np.isfinite(value) or rel > 2e-5:
        raise CalibrationError(f"Analytic gradient test failed: relative difference={rel}")
    fit = fit_model(batch, True, config)
    p = fit["parameters"]
    tolerances = {"p_init": 0.10, "p_learn": 0.08, "slip": 0.06, "guess": 0.06, "lambda_per_day": 0.04}
    for name, tol in tolerances.items():
        if abs(float(p[name]) - float(true.as_dict()[name])) > tol:
            raise CalibrationError(f"Synthetic recovery outside tolerance for {name}: {p[name]} vs {true.as_dict()[name]}")
    probs, yy, gg = predict_probabilities(model_natural_array(fit), batch, True)
    metrics = prediction_metrics(yy, probs, 10)
    if metrics["log_loss"] >= 0.70 or probs.size != batch.n_events or gg.size != batch.n_events:
        raise CalibrationError("Synthetic prediction test failed")
    print("SELF-TEST: PASS")
    print(json.dumps({"gradient_relative_error": rel, "recovered": p, "metrics": metrics}, indent=2, allow_nan=True))


def cleanup_tree(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    last = None
    for attempt in range(8):
        try:
            shutil.rmtree(path)
            return
        except Exception as exc:
            last = exc
            time.sleep(0.15 * (attempt + 1))
    print(f"WARNING: temporary directory could not be removed: {path}: {last}", file=sys.stderr)


def run_pipeline(input_bundle: Path, output_dir: Path, config: FitConfig) -> Path:
    t0 = time.perf_counter()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_input: Path | None = None
    try:
        root, temp_input = locate_bundle_root(input_bundle.resolve(), output_dir.parent)
        print(f"INPUT ROOT: {root}", flush=True)
        manifest, validation, input_identity = verify_bundle(root)
        validate_parquet_headers(root, manifest, validation)
        config_identity = hashlib.sha256(canonical_json_bytes(config.as_dict())).hexdigest()
        run_identity = hashlib.sha256((input_identity + config_identity).encode("ascii")).hexdigest()
        work = output_dir / ".work"
        work.mkdir(parents=True, exist_ok=True)
        identity_path = work / "RUN_IDENTITY.json"
        identity_obj = {"run_identity_hash": run_identity, "input_identity_hash": input_identity, "config": config.as_dict()}
        if identity_path.is_file() and read_json(identity_path) != identity_obj:
            raise CalibrationError("Output .work belongs to a different input/configuration. Use another OutputDir or remove .work deliberately.")
        atomic_json(identity_path, identity_obj)
        print("BUNDLE VALIDATION: PASS", flush=True)
        arrays = load_reference_arrays(root, manifest)
        print(f"REFERENCE LOAD: {arrays['outcome'].size:,} rows", flush=True)
        validate_split_and_pair_consistency(root, arrays, manifest)
        print("REFERENCE CONSISTENCY: PASS", flush=True)
        results = []
        for skill_id in [int(x) for x in manifest["selection"]["selected_skills"]]:
            results.append(fit_one_skill(skill_id, arrays, config, run_identity, work))
        results.sort(key=lambda r: int(r["skill_id"]))
        final_dir = output_dir / "results"
        if final_dir.exists():
            shutil.rmtree(final_dir)
        zip_path = build_output_bundle(final_dir, input_identity, config, manifest, results, time.perf_counter() - t0)
        atomic_json(output_dir / "COMPLETE.json", {
            "status": "PASS", "run_identity_hash": run_identity,
            "created_at_utc": utc_iso(), "result_zip": str(zip_path), "result_zip_sha256": sha256_file(zip_path),
        })
        print("CALIBRATION: PASS", flush=True)
        print(f"UPLOAD: {zip_path}", flush=True)
        return zip_path
    finally:
        cleanup_tree(temp_input)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fit per-skill BKT and BKT-F models to the AdaptiveLearningSim KT1 reference bundle.")
    p.add_argument("--input-bundle", type=Path, help="Reference bundle ZIP or extracted directory")
    p.add_argument("--output-dir", type=Path, help="Output directory")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--starts-bkt", type=int, default=8)
    p.add_argument("--starts-bktf", type=int, default=12)
    p.add_argument("--maxiter", type=int, default=350)
    p.add_argument("--seed", type=int, default=20260731)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.self_test:
            run_self_test()
            return 0
        if args.input_bundle is None or args.output_dir is None:
            raise CalibrationError("--input-bundle and --output-dir are required unless --self-test is used")
        if args.starts_bkt < 2 or args.starts_bktf < 2 or args.maxiter < 50:
            raise CalibrationError("Invalid optimization settings")
        config = FitConfig(starts_bkt=args.starts_bkt, starts_bktf=args.starts_bktf, maxiter=args.maxiter, seed=args.seed)
        run_pipeline(args.input_bundle, args.output_dir, config)
        return 0
    except KeyboardInterrupt:
        print("INTERRUPTED: completed per-skill checkpoints were preserved.", file=sys.stderr)
        return 130
    except CalibrationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
