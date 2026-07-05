set -euo pipefail

# Usage:
#   bash scripts/infer_tusam.sh
#   CKPT=best bash scripts/infer_tusam.sh
#   CONFIG=configs/tusam.yaml CKPT=last bash scripts/infer_tusam.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${CONFIG:-${ROOT_DIR}/configs/tusam.yaml}"
MODEL_NAME="$(CONFIG_PATH="${CONFIG}" python - <<'PY'
import os, yaml
p=os.environ['CONFIG_PATH']
with open(p,'r',encoding='utf-8') as f:
    cfg=yaml.safe_load(f)
print(str(cfg.get('student_model','pvt')))
PY
)"
WORK_DIR="${WORK_DIR:-${ROOT_DIR}/outputs/tusam/${MODEL_NAME}}"
LOG_DIR="${WORK_DIR}"
CKPT="${CKPT:-best}"   # best | last
SAVE_DIR="${SAVE_DIR:-${WORK_DIR}/infer_${CKPT}}"

mkdir -p "${LOG_DIR}"
mkdir -p "${SAVE_DIR}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/segment-anything:${PYTHONPATH:-}"

if [[ "${CKPT}" == "best" ]]; then
  CKPT_PATH="${WORK_DIR}/best.pth"
elif [[ "${CKPT}" == "last" ]]; then
  CKPT_PATH="${WORK_DIR}/last.pth"
else
  echo "[ERROR] CKPT must be 'best' or 'last', got: ${CKPT}"
  exit 1
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "[ERROR] checkpoint not found: ${CKPT_PATH}"
  exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] config not found: ${CONFIG}"
  exit 1
fi

echo "[INFO] Config: ${CONFIG}"
echo "[INFO] Work dir: ${WORK_DIR}"
echo "[INFO] Checkpoint: ${CKPT_PATH}"
echo "[INFO] Save dir: ${SAVE_DIR}"

CONFIG="${CONFIG}" CKPT_PATH="${CKPT_PATH}" SAVE_DIR="${SAVE_DIR}" python - <<'PY' 2>&1 | tee "${LOG_DIR}/infer_${CKPT}.log"
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_tusam import compute_metrics, infer_tta
from tusam.data.dataset import TestDataset
from tusam.models.student_factory import build_student
from tusam.utils.config import load_cfg

config_path = os.environ["CONFIG"]
ckpt_path = os.environ["CKPT_PATH"]
save_dir = Path(os.environ["SAVE_DIR"])

cfg = load_cfg(config_path)
device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

student_name = str(getattr(cfg, "student_model", "pvt"))
model = build_student(student_name).to(device)
ck = torch.load(ckpt_path, map_location=device)
if isinstance(ck, dict) and "student" in ck:
    model.load_state_dict(ck["student"], strict=True)
else:
    model.load_state_dict(ck, strict=False)
model.eval()
print(f"[Infer] student_model={student_name}")

root = Path(cfg.test_root)
results = {}

for ds in sorted([d for d in os.listdir(root) if (root / d).is_dir()]):
    ds_root = root / ds
    loader = DataLoader(TestDataset(str(ds_root), cfg.image_size), batch_size=8, shuffle=False, num_workers=4)

    out_mask_dir = save_dir / ds / "masks"
    out_overlay_dir = save_dir / ds / "overlays"
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    out_overlay_dir.mkdir(parents=True, exist_ok=True)

    dices, ious = [], []
    pbar = tqdm(loader, desc=f"Infer {ds}")
    for batch in pbar:
        img = batch["image"].to(device)
        gt = batch["mask"].to(device)
        names = batch["name"]

        pred = infer_tta(model, img, full=True, post_process=True)
        d, i = compute_metrics(pred, gt)
        dices.append(d)
        ious.append(i)

        pred_np = pred.detach().cpu().numpy()  # [B,1,H,W], already binary float
        img_np = img.detach().cpu().numpy()    # [B,3,H,W], [0,1]

        for bi in range(pred_np.shape[0]):
            name = names[bi]
            pm = (pred_np[bi, 0] > 0.5).astype(np.uint8) * 255
            im = (np.transpose(img_np[bi], (1, 2, 0)) * 255.0).astype(np.uint8)

            cv2.imwrite(str(out_mask_dir / name), pm)

            overlay = im.copy()
            m = pm > 127
            overlay[m] = (0.6 * overlay[m] + 0.4 * np.array([255, 80, 80])).astype(np.uint8)
            contours, _ = cv2.findContours((pm > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
            cv2.imwrite(str(out_overlay_dir / name), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        pbar.set_postfix(mDice=f"{float(np.mean(dices)):.4f}", mIoU=f"{float(np.mean(ious)):.4f}")

    results[ds] = {"mDice": float(np.mean(dices)), "mIoU": float(np.mean(ious))}

results["Overall"] = {
    "mDice": float(np.mean([v["mDice"] for v in results.values()])),
    "mIoU": float(np.mean([v["mIoU"] for v in results.values()])),
}

print("\n[Inference Results]")
print(results)
print(f"\nSaved masks/overlays to: {save_dir}")
PY

echo "[INFO] Inference done. Log: ${LOG_DIR}/infer_${CKPT}.log"
echo "[INFO] Outputs saved in: ${SAVE_DIR}"
