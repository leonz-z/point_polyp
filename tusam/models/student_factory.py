from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tusam.models.pvt_fpn_student import PVTv2FPNStudent


class SingleMapStudentWrapper(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x):
        out = self.base(x)
        if isinstance(out, (list, tuple)):
            main = out[0]
            aux1 = out[1] if len(out) > 1 else out[0]
            aux2 = out[2] if len(out) > 2 else out[0]
        else:
            main = out
            aux1 = out
            aux2 = out

        if main.ndim == 4 and main.shape[1] == 2:
            main = main[:, 1:2]
        if aux1.ndim == 4 and aux1.shape[1] == 2:
            aux1 = aux1[:, 1:2]
        if aux2.ndim == 4 and aux2.shape[1] == 2:
            aux2 = aux2[:, 1:2]

        h, w = x.shape[-2:]
        main = F.interpolate(main, size=(h, w), mode="bilinear", align_corners=False)
        aux1 = F.interpolate(aux1, size=(h, w), mode="bilinear", align_corners=False)
        aux2 = F.interpolate(aux2, size=(h, w), mode="bilinear", align_corners=False)
        return main, aux1, aux2


def build_student(name: str) -> nn.Module:
    n = name.lower()
    if n in ("pvt", "polyp-pvt", "polyp_pvt", "pvt-fpn"):
        return PVTv2FPNStudent()

    if n == "unet":
        from TextPolyp.lib.unet import UNet

        return SingleMapStudentWrapper(UNet(in_chns=3, class_num=1))

    if n in ("res2net", "net"):
        from TextPolyp.lib.Net import Net

        return SingleMapStudentWrapper(Net(pretrained_backbone=False))

    if n == "pranet":
        from TextPolyp.lib.PraNet_Res2Net import PraNet

        return SingleMapStudentWrapper(PraNet(pretrained_backbone=False))

    raise ValueError(f"Unsupported student model: {name}")


def load_student_backbone_if_needed(model: nn.Module, model_name: str, pvt_ckpt: str | None = None):
    n = model_name.lower()
    if n in ("pvt", "polyp-pvt", "polyp_pvt", "pvt-fpn") and pvt_ckpt:
        if hasattr(model, "backbone"):
            ckpt = torch.load(pvt_ckpt, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]

            # Drop classification head weights from ImageNet pretraining checkpoints.
            filtered = {}
            removed = []
            for k, v in ckpt.items():
                if k.startswith("head.") or k.startswith("classifier.") or k.startswith("fc."):
                    removed.append(k)
                    continue
                filtered[k] = v

            miss = model.backbone.load_state_dict(filtered, strict=False)
            msg = f"Loaded local PVT checkpoint: {pvt_ckpt}"
            if removed:
                msg += f"; dropped classifier keys: {len(removed)}"
            msg += f"; missing/unexpected: {miss}"
            return msg
    return "No extra backbone checkpoint loaded."
