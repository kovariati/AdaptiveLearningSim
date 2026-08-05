#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FITS="${1:-$HERE/skill_model_fits.csv}"
ITEMS="${2:-$HERE/selected_items.csv}"
OUT="${3:-$HERE/rebuilt_inputs}"
mkdir -p "$OUT"
python "$HERE/estimate_empirical_bayes_hyperparameters.py" \
  --skill-fits "$FITS" \
  --out-json "$OUT/empirical_bayes_hyperparameters.json" \
  --out-audit-csv "$OUT/empirical_bayes_hyperparameter_audit.csv"
python "$HERE/build_empirical_calibration_inputs.py" \
  --skill-fits "$FITS" \
  --selected-items "$ITEMS" \
  --config "$OUT/empirical_bayes_hyperparameters.json" \
  --out-dir "$OUT"
sha256sum "$OUT/input_skill_parameter_shrinkage.csv" "$OUT/input_item_effect_audit.csv" > "$OUT/SHA256SUMS.txt"
