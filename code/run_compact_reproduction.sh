#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$ROOT/../reproduced_results}"
MODE="${2:-quick}"
if [[ "$MODE" == "quick" ]]; then
  REPS=1; LEARNERS=25; STEPS=12; ORDERS=2; RESP=3000; SREPS=1; SLEARN=30; SORDERS=2; PDRAWS=3; PMC=2; PLEARN=20; PBOOT=50
else
  REPS=30; LEARNERS=250; STEPS=100; ORDERS=5; RESP=20000; SREPS=10; SLEARN=200; SORDERS=3; PDRAWS=200; PMC=3; PLEARN=40; PBOOT=2000
fi
python "$ROOT/run_taskenv.py" \
  --skill-parameters "$ROOT/input_skill_parameter_shrinkage.csv" \
  --item-effects "$ROOT/input_item_effect_audit.csv" \
  --output-dir "$OUT" \
  --replicates "$REPS" --learners "$LEARNERS" --practice-steps "$STEPS" --delayed-days 30 \
  --skill-orders "$ORDERS" --response-diagnostic-learners "$RESP" \
  --sensitivity-replicates "$SREPS" --sensitivity-learners "$SLEARN" --sensitivity-orders "$SORDERS" \
  --parameter-draws "$PDRAWS" --parameter-mc-replicates "$PMC" --parameter-learners "$PLEARN" --parameter-tail-bootstrap "$PBOOT"
if [[ "${ALS_RUN_TESTS:-1}" == "1" ]]; then
  python -m pytest -q "$ROOT/test_taskenv.py"
fi
