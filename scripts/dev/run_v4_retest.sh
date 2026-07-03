#!/usr/bin/env bash
# Retest H2 + alertes on ewat_v4 after collect/build/assemble.
# Prerequisite: data/datasets/ewat_v4 exists (see docs/runbook_v4.md).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATASET="${DATASET:-data/datasets/ewat_v4}"
FEATURES="${FEATURES:-data/features/v4}"
OUT="${OUT:-experiments/v4_retest}"
SEED="${SEED:-42}"

if [[ ! -f "$DATASET/split.json" ]]; then
  echo "Missing $DATASET/split.json — run assemble first (docs/runbook_v4.md)." >&2
  echo "ewat_v4 not collected: document as future work (docs/ewat_v4_decision.md)." >&2
  exit 1
fi

mkdir -p "$OUT"

echo "=== Calibrate drift (v4) ==="
python -m experiments.drift_separation.calibrate \
  --dataset "$DATASET" \
  --features-root "$FEATURES" \
  --output "$OUT/drift"

echo "=== Train encoder (seed=$SEED) ==="
python -m experiments.encoder.train \
  --dataset "$DATASET" \
  --features-root "$FEATURES" \
  --output "$OUT/encoder" \
  --epochs 100 \
  --seed "$SEED"

echo "=== Train typing ==="
python -m experiments.typing.train \
  --dataset "$DATASET" \
  --features-root "$FEATURES" \
  --encoder-checkpoint "$OUT/encoder/checkpoints/best_encoder.pt" \
  --output "$OUT/typing" \
  --epochs 50 \
  --seed "$SEED"

echo "=== Train precursors ==="
python -m experiments.precursor.train \
  --typing-dir "$OUT/typing" \
  --features-root "$FEATURES" \
  --output "$OUT/precursor" \
  --k-values 2 4 6 8 10 12 14 16

echo "=== H2 look-through (calibrate ε on v4 train + test eval) ==="
python -m experiments.h2_lookthrough.run \
  --dataset "$DATASET" \
  --output "$OUT/h2_lookthrough"

echo "=== Alerts (threshold 0.7 + ROC) ==="
python -m experiments.alerts.eval \
  --typing-dir "$OUT/typing" \
  --encoder-dir "$OUT/encoder" \
  --precursor-dir "$OUT/precursor" \
  --features-root "$FEATURES" \
  --output "$OUT/alerts" \
  --p-thresholds 0.3 0.4 0.5 0.6 0.7 \
  --roc-sweep \
  --n-bootstrap 1000

python -m scripts.export_thesis_figures \
  --typing-dir "$OUT/typing" \
  --encoder-dir "$OUT/encoder" \
  --precursor-dir "$OUT/precursor" \
  --features-root "$FEATURES" \
  --alerts-dir "$OUT/alerts" \
  --skip-eval

echo "Done. Results under $OUT/"
