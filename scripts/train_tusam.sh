#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
set -euo pipefail

# Usage:
#   bash scripts/train_tusam.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train_tusam.sh

ROOT_DIR=""
CONFIG="${ROOT_DIR}/configs/tusam.yaml"
MODEL_NAME="$(python - <<'PY'
import yaml
with open('','r',encoding='utf-8') as f:
    cfg=yaml.safe_load(f)
print(str(cfg.get('student_model','pvt')))
PY
)"
LOG_DIR="${ROOT_DIR}/outputs/tusam/${MODEL_NAME}"
mkdir -p "${LOG_DIR}"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/segment-anything:${PYTHONPATH:-}"

python "${ROOT_DIR}/train_tusam.py" \
  --config "${CONFIG}" \
  2>&1 | tee "${LOG_DIR}/train.log"
