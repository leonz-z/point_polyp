from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PointTrainDataset(Dataset):
    def __init__(self, train_root: str, point_root: str, image_size: int = 352):
        self.train_root = Path(train_root)
        self.point_root = Path(point_root)
        self.image_root = self.train_root / "images"
        self.image_size = image_size

        raw_ids = sorted([p.stem for p in self.point_root.glob("*.json")], key=lambda x: int(x))
        self.ids = []
        self.invalid_ids = []
        for sid in raw_ids:
            point_path = self.point_root / f"{sid}.json"
            fg, _, _, _ = self._load_points(point_path)
            if fg is None:
                self.invalid_ids.append(sid)
            else:
                self.ids.append(sid)

        if len(self.invalid_ids) > 0:
            print(
                f"[PointTrainDataset] Skip {len(self.invalid_ids)} samples without foreground point: "
                f"{', '.join(self.invalid_ids[:10])}"
            )

    def __len__(self) -> int:
        return len(self.ids)

    def _load_points(self, point_file: Path):
        with open(point_file, "r", encoding="utf-8") as f:
            ann = json.load(f)

        fg = None
        bg = []
        for s in ann.get("shapes", []):
            if "points" not in s or len(s["points"]) == 0:
                continue
            p = s["points"][0]
            label = str(s.get("label", "")).strip().lower()
            if label in ("foreground", "fg", "polyp", "lesion", "positive", "pos"):
                fg = p
            elif label in ("background", "bg", "negative", "neg"):
                bg.append(p)

        h, w = ann["imageHeight"], ann["imageWidth"]
        return fg, bg, h, w

    def __getitem__(self, idx: int):
        sid = self.ids[idx]
        image_path = self.image_root / f"{sid}.png"
        point_path = self.point_root / f"{sid}.json"

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fg, bg, h0, w0 = self._load_points(point_path)

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        sx = self.image_size / float(w0)
        sy = self.image_size / float(h0)

        if fg is None:
            raise ValueError(f"No foreground point found in annotation: {point_path}")

        fg = np.array([fg[0] * sx, fg[1] * sy], dtype=np.float32)
        if len(bg) > 0:
            bg = np.array([[p[0] * sx, p[1] * sy] for p in bg], dtype=np.float32)
        else:
            bg = np.zeros((0, 2), dtype=np.float32)

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        return {
            "id": sid,
            "image": torch.from_numpy(image),
            "fg_point": torch.from_numpy(fg),
            "bg_points": torch.from_numpy(bg),
        }


class TestDataset(Dataset):
    def __init__(self, ds_root: str, image_size: int = 352):
        self.ds_root = Path(ds_root)
        self.image_root = self.ds_root / "images"
        self.mask_root = self.ds_root / "masks"
        self.image_size = image_size
        self.files = sorted([p.name for p in self.image_root.glob("*.png")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        img = cv2.imread(str(self.image_root / fn), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_root / fn), cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        mask = (mask > 127).astype(np.float32)

        return {
            "image": torch.from_numpy(img),
            "mask": torch.from_numpy(mask[None]),
            "name": fn,
        }
