#!/usr/bin/env bash
set -euo pipefail

CASE_ROOT="${CASE_ROOT:-/public_bme2/bme-cuizhm/maxquan/Datasets/CBCT/2d_projection_physics_consistent}"
CASE_NAMES_FILE="${CASE_NAMES_FILE:-tiny_ablation/case_splits/tiny_cases.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-tiny_ablation/outputs}"
NUM_CASES="${NUM_CASES:-16}"
RUN_TAG="${RUN_TAG:-branch_$(date +%Y%m%d_%H%M%S)}"

echo "Branch tiny run tag: ${RUN_TAG}"
echo "Outputs will be written under: ${OUTPUT_ROOT}/${RUN_TAG}"

python tiny_ablation/scripts/make_tiny_cases.py \
  --case-root "${CASE_ROOT}" \
  --out "${CASE_NAMES_FILE}" \
  --num-cases "${NUM_CASES}"

python tiny_ablation/scripts/run_branch_tiny_ablation.py \
  --case-root "${CASE_ROOT}" \
  --case-names-file "${CASE_NAMES_FILE}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-tag "${RUN_TAG}" \
  "$@"
