#!/usr/bin/env python3
"""Reassemble and verify the derived EdNet-KT1 reference bundle."""
from __future__ import annotations
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "AdaptiveLearningSim_KT1_reference_bundle.zip"
PARTS = sorted(HERE.glob("AdaptiveLearningSim_KT1_reference_bundle.part*"))
EXPECTED = (HERE / "REFERENCE_BUNDLE_SHA256.txt").read_text(encoding="utf-8").split()[0].lower()
if not PARTS:
    raise SystemExit("No reference-bundle parts were found.")
h = hashlib.sha256()
with OUT.open("wb") as dst:
    for part in PARTS:
        block = part.read_bytes()
        dst.write(block)
        h.update(block)
actual = h.hexdigest().lower()
if actual != EXPECTED:
    OUT.unlink(missing_ok=True)
    raise SystemExit(f"SHA-256 mismatch: expected {EXPECTED}, got {actual}")
print(f"Created {OUT.name}")
print(f"SHA-256 {actual}")
