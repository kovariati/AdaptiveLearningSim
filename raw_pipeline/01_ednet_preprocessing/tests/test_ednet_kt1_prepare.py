from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "ednet_kt1_prepare.py"
spec = importlib.util.spec_from_file_location("ednet_kt1_prepare", MODULE_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def write_questions(path: Path) -> None:
    path.write_text(
        "question_id,bundle_id,explanation_id,correct_answer,part,tags,deployed_at\n"
        "q1,b1,e1,a,1,10,1000\n"
        "q2,b2,e2,b,2,20;21,2000\n"
        "q3,b3,e3,c,3,-1,-1\n",
        encoding="utf-8",
    )


def write_user(path: Path, rows: list[tuple]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(mod.REQUIRED_KT1_COLUMNS)
        writer.writerows(rows)


def test_real_contents_zip_loads() -> None:
    zip_path = Path("/mnt/data/EdNet-Contents.zip")
    if not zip_path.exists():
        pytest.skip("conversation attachment unavailable")
    metadata, summary = mod.load_question_metadata(zip_path)
    assert len(metadata) == 13169
    assert summary["unique_questions"] == 13169
    assert metadata["q1"].correct_answer == "b"
    assert metadata["q1"].skill_ids == "1;2;179;181"
    assert metadata["q1"].skill_count == 4


def test_parse_flags_and_stable_sort(tmp_path: Path) -> None:
    q = tmp_path / "questions.csv"
    write_questions(q)
    metadata, _ = mod.load_question_metadata(q)
    u = tmp_path / "u7.csv"
    write_user(
        u,
        [
            (3000, 1, "q1", "a", 100),
            (2500, 1, "q2", "b", 200),
            (2500, 1, "q2", "b", 200),
            (4000, 2, "q3", "z", -1),
            (5000, 3, "q999", "a", "bad"),
        ],
    )
    result = mod.parse_learner_file(u, metadata, extreme_elapsed_ms=150)
    assert result.error_record is None
    assert [r[2] for r in result.interactions] == [2500, 2500, 3000, 4000, 5000]
    assert result.learner_summary[8] == 1
    assert result.learner_summary[9] == 1
    assert result.learner_summary[10] == 1
    assert result.learner_summary[11] == 1
    assert result.learner_summary[13] == 1
    assert result.learner_summary[14] == 2
    assert result.interactions[1][-1] & mod.Q_DUPLICATE_EVENT
    assert result.interactions[-1][-1] & mod.Q_UNKNOWN_QUESTION
    assert result.interactions[-1][-1] & mod.Q_INVALID_ELAPSED


def test_csv_gzip_atomic_roundtrip(tmp_path: Path) -> None:
    rows = [(1, "a"), (2, None)]
    out = tmp_path / "x.csv.gz"
    mod.write_csv_gzip_atomic(out, ("n", "s"), rows)
    with gzip.open(out, "rt", encoding="utf-8", newline="") as fh:
        data = list(csv.reader(fh))
    assert data == [["n", "s"], ["1", "a"], ["2", ""]]


def test_full_small_run_and_resume(tmp_path: Path) -> None:
    q = tmp_path / "questions.csv"
    write_questions(q)
    kt1 = tmp_path / "KT1"
    kt1.mkdir()
    write_user(kt1 / "u1.csv", [(1000, 1, "q1", "a", 50), (2000, 2, "q2", "a", 60)])
    write_user(kt1 / "u2.csv", [(1000, 1, "q3", "c", 70)])
    write_user(kt1 / "u3.csv", [(1000, 1, "q1", "d", 80)])
    output = tmp_path / "out"

    argv = [
        "--kt1-dir", str(kt1),
        "--contents", str(q),
        "--output-dir", str(output),
        "--output-format", "csv-gzip",
        "--files-per-shard", "2",
        "--workers", "2",
        "--preflight-sample-files", "3",
        "--skip-disk-check",
    ]
    assert mod.main(argv) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["output"]["learner_rows"] == 3
    assert manifest["output"]["interaction_rows"] == 4
    assert manifest["output"]["shard_count"] == 2
    before = {
        p.name: mod.sha256_file(p)
        for p in (output / "interactions").glob("*.csv.gz")
    }
    assert mod.main(argv) == 0
    after = {
        p.name: mod.sha256_file(p)
        for p in (output / "interactions").glob("*.csv.gz")
    }
    assert before == after


def test_resume_identity_rejects_changed_shard_size(tmp_path: Path) -> None:
    q = tmp_path / "questions.csv"
    write_questions(q)
    kt1 = tmp_path / "KT1"
    kt1.mkdir()
    write_user(kt1 / "u1.csv", [(1000, 1, "q1", "a", 50)])
    write_user(kt1 / "u2.csv", [(1000, 1, "q2", "b", 50)])
    write_user(kt1 / "u3.csv", [(1000, 1, "q3", "c", 50)])
    output = tmp_path / "out"
    common = [
        "--kt1-dir", str(kt1), "--contents", str(q), "--output-dir", str(output),
        "--output-format", "csv-gzip", "--workers", "1",
        "--preflight-sample-files", "3", "--skip-disk-check",
    ]
    assert mod.main(common + ["--files-per-shard", "2"]) == 0
    assert mod.main(common + ["--files-per-shard", "3"]) == 1


def test_internal_selftest_csv() -> None:
    result = mod.run_internal_self_tests("csv-gzip")
    assert result["status"] == "PASS"
