#!/bin/bash
set -euo pipefail

CASE_ROOT="${CASE_ROOT:-/public_bme2/bme-cuizhm/maxquan/Datasets/CBCT/2d_projection_physics_consistent}"
CASE_LIST="${CASE_LIST:-tiny_ablation/case_splits/tiny_cases.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-tiny_ablation/outputs}"

python tiny_ablation/scripts/make_tiny_cases.py \
  --case-root "$CASE_ROOT" \
  --out "$CASE_LIST" \
  --num-cases "${NUM_CASES:-16}"

python tiny_ablation/scripts/run_tiny_ablation.py \
  --case-root "$CASE_ROOT" \
  --case-names-file "$CASE_LIST" \
  --output-root "$OUTPUT_ROOT"
