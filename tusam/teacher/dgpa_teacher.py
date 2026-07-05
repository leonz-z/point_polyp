from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything import SamPredictor, sam_model_registry


class DGPA(nn.Module):
    def __init__(self, c_in: int = 256, sigma_min: float = 0.02, sigma_max: float = 0.25):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.mlp = nn.Sequential(
            nn.Linear(c_in, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, feat_vec: torch.Tensor):
        s = torch.sigmoid(self.mlp(feat_vec))
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * s
        return sigma.squeeze(-1)


@dataclass
class PseudoPack:
    soft_mask: torch.Tensor
    hard_mask: torch.Tensor
    quality: torch.Tensor
    cand_soft_masks: torch.Tensor | None = None
    prm_before: torch.Tensor | None = None
    prm_after: torch.Tensor | None = None


class FrozenSamTeacher:
    def __init__(self, sam_type: str, sam_ckpt: str, device: str = "cuda", sigma_min: float = 0.02, sigma_max: float = 0.25, vis_cb=None):
        self.device = device
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt)
        sam.to(device=device)
        sam.eval()
        for p in sam.parameters():
            p.requires_grad = False
        self.predictor = SamPredictor(sam)

        c_in = int(getattr(sam.prompt_encoder, "embed_dim", 256))
        self.dgpa = DGPA(c_in=c_in, sigma_min=sigma_min, sigma_max=sigma_max).to(device)
        self.vis_cb = vis_cb

    @staticmethod
    def _safe_sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _image_to_uint8(image_t: torch.Tensor) -> np.ndarray:
        return (image_t.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

    def _sam_embed_and_point_on_feature(self, image_t: torch.Tensor, fg_point_t: torch.Tensor):
        image = self._image_to_uint8(image_t)
        original_size = image.shape[:2]

        transformed = self.predictor.transform.apply_image(image)
        transformed_t = torch.as_tensor(transformed, device=self.device).permute(2, 0, 1).contiguous()[None]

        with torch.no_grad():
            self.predictor.set_torch_image(transformed_t, original_size)
            emb = self.predictor.get_image_embedding()  # [1,C,Hf,Wf]

        _, c, hf, wf = emb.shape

        pt_np = fg_point_t.detach().cpu().numpy().astype(np.float32)[None, :]
        pt_tfm = self.predictor.transform.apply_coords(pt_np, original_size)[0]  # in SAM resized input frame

        img_size = float(self.predictor.model.image_encoder.img_size)
        x_f = float(np.clip(pt_tfm[0], 0.0, img_size - 1.0)) / max(1.0, img_size - 1.0) * (wf - 1)
        y_f = float(np.clip(pt_tfm[1], 0.0, img_size - 1.0)) / max(1.0, img_size - 1.0) * (hf - 1)

        grid_x = (x_f / max(1, wf - 1)) * 2 - 1
        grid_y = (y_f / max(1, hf - 1)) * 2 - 1
        grid = torch.tensor([[[[grid_x, grid_y]]]], device=self.device, dtype=emb.dtype)

        feat = F.grid_sample(emb, grid, mode="bilinear", align_corners=True).view(1, c)
        return feat

    def predict_sigma_batch(self, images: torch.Tensor, fg_points: torch.Tensor) -> torch.Tensor:
        feats = []
        for i in range(images.shape[0]):
            feats.append(self._sam_embed_and_point_on_feature(images[i], fg_points[i]))
        feat = torch.cat(feats, dim=0)
        sigma = self.dgpa(feat)
        return sigma

    @torch.no_grad()
    def predict_sigma_single(self, image_t: torch.Tensor, fg_point_t: torch.Tensor) -> float:
        feat = self._sam_embed_and_point_on_feature(image_t, fg_point_t)
        s = self.dgpa(feat)[0]
        return float(s.detach().cpu().item())

    @torch.no_grad()
    def _run_sam_multi(self, image: np.ndarray, fg_point: np.ndarray, bg_points: np.ndarray, mask_input: np.ndarray):
        self.predictor.set_image(image)
        pts = [fg_point.astype(np.float32)]
        lbs = [1]
        if bg_points.shape[0] > 0:
            pts += [p.astype(np.float32) for p in bg_points]
            lbs += [0] * bg_points.shape[0]

        point_coords = np.stack(pts, axis=0)
        point_labels = np.array(lbs, dtype=np.int32)

        masks, ious, low_res = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input[None, :, :],
            multimask_output=True,
            return_logits=True,
        )
        return masks, ious, low_res

    def _stability(self, logit: np.ndarray):
        bins = [0.3, 0.4, 0.5, 0.6, 0.7]
        areas = []
        for t in bins:
            areas.append((logit > t).sum() + 1e-6)
        areas = np.array(areas, dtype=np.float32)
        return float(np.min(areas) / np.max(areas))

    def _refine_mask(self, mask: np.ndarray, fg_point: np.ndarray):
        h, w = mask.shape
        n, cc, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        keep = np.zeros_like(mask, dtype=np.uint8)
        x, y = int(np.clip(fg_point[0], 0, w - 1)), int(np.clip(fg_point[1], 0, h - 1))
        target = cc[y, x]
        if target > 0:
            keep[cc == target] = 1
        else:
            if n > 1:
                sizes = st[1:, cv2.CC_STAT_AREA]
                keep[cc == (1 + int(np.argmax(sizes)))] = 1

        area_ratio = keep.mean()
        k = 3 if area_ratio < 0.02 else 5
        ker = np.ones((k, k), np.uint8)
        keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, ker)
        keep = cv2.morphologyEx(keep, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        keep = cv2.medianBlur(keep, 3)
        keep = (keep > 0).astype(np.uint8)
        return keep

    def generate_pseudo(self, image_t: torch.Tensor, fg_point_t: torch.Tensor, bg_points_t: torch.Tensor, sigma: float | None, sample_id: str | None = None):
        image = self._image_to_uint8(image_t)
        h, w = image.shape[:2]
        fg = fg_point_t.cpu().numpy()
        bg = bg_points_t.cpu().numpy()

        if sigma is None:
            sigma = self.predict_sigma_single(image_t, fg_point_t)

        ys, xs = np.mgrid[0:h, 0:w]
        g = np.exp(-((xs - fg[0]) ** 2 + (ys - fg[1]) ** 2) / (2 * (max(1e-6, sigma * max(h, w))) ** 2 + 1e-6))

        # Construct a robust dense prompt from the Gaussian prior.
        # 1) sharpen near the foreground center,
        # 2) suppress weak responses in the tail,
        # 3) keep the prompt smooth for SAM mask_input.
        dense = np.power(np.clip(g, 0.0, 1.0), 1.6).astype(np.float32)
        dense = cv2.GaussianBlur(dense, (0, 0), sigmaX=max(1.0, sigma * max(h, w) * 0.5), sigmaY=max(1.0, sigma * max(h, w) * 0.5))
        dense = np.clip(dense, 0.0, 1.0)
        dense = np.where(dense > 0.12, dense, 0.0).astype(np.float32)
        dense = dense / max(1e-6, float(dense.max()))
        mask_input_soft = cv2.resize(dense.astype(np.float32), (256, 256), interpolation=cv2.INTER_LINEAR)
        mask_input = cv2.resize((dense * 2.0 - 1.0).astype(np.float32), (256, 256), interpolation=cv2.INTER_LINEAR)
        if self.vis_cb is not None:
            self.vis_cb("dgpa_prompt", sample_id=sample_id, image=image, fg=fg, bg=bg, sigma=float(sigma), gaussian=g, mask_input=mask_input, mask_input_soft=mask_input_soft)

        masks, ious, lows = self._run_sam_multi(image, fg, bg, mask_input)
        if self.vis_cb is not None:
            self.vis_cb("sam_multimask", sample_id=sample_id, image=image, fg=fg, bg=bg, masks=masks, ious=ious, lows=lows, mask_input=mask_input)

        best_score, best_idx = -1e9, 0
        for k in range(len(masks)):
            mlogit = self._safe_sigmoid(lows[k])
            mprob = cv2.resize(mlogit, (w, h), interpolation=cv2.INTER_LINEAR)
            vio = 0
            for p in bg:
                x, y = int(np.clip(p[0], 0, w - 1)), int(np.clip(p[1], 0, h - 1))
                if mprob[y, x] > 0.5:
                    vio += 1
            stab = self._stability(mprob)
            score = float(ious[k]) * stab * math.exp(-1.2 * vio)
            if score > best_score:
                best_score, best_idx = score, k

        cand_soft = []
        for k in range(len(lows)):
            ck = self._safe_sigmoid(lows[k])
            ck = cv2.resize(ck, (w, h), interpolation=cv2.INTER_LINEAR)
            cand_soft.append(ck.astype(np.float32))

        best_prob = self._safe_sigmoid(lows[best_idx])
        best_prob = cv2.resize(best_prob, (w, h), interpolation=cv2.INTER_LINEAR)
        hard_before = (best_prob > 0.5).astype(np.uint8)
        hard = self._refine_mask(hard_before, fg)
        if self.vis_cb is not None:
            self.vis_cb("prm_refine", sample_id=sample_id, image=image, fg=fg, bg=bg, hard_before=hard_before, hard_after=hard, best_prob=best_prob, cand_soft=cand_soft)

        if hard.mean() < 0.005:
            core = np.argwhere(best_prob > 0.8)
            ring = np.argwhere((best_prob > 0.2) & (best_prob < 0.4))
            add_fg = core[np.random.choice(len(core), min(2, len(core)), replace=False)] if len(core) > 0 else np.zeros((0, 2), int)
            add_bg = ring[np.random.choice(len(ring), min(3, len(ring)), replace=False)] if len(ring) > 0 else np.zeros((0, 2), int)
            if len(add_fg) > 0 or len(add_bg) > 0:
                fg2 = np.array([float(add_fg[0][1]), float(add_fg[0][0])], dtype=np.float32) if len(add_fg) > 0 else fg
                bg2 = bg
                if len(add_bg) > 0:
                    add_bg_xy = np.stack([add_bg[:, 1], add_bg[:, 0]], axis=1).astype(np.float32)
                    bg2 = np.concatenate([bg, add_bg_xy], axis=0) if bg.shape[0] else add_bg_xy
                masks2, ious2, lows2 = self._run_sam_multi(image, fg2, bg2, mask_input)
                if float(np.max(ious2)) > float(ious[best_idx]):
                    j = int(np.argmax(ious2))
                    best_prob = self._safe_sigmoid(lows2[j])
                    best_prob = cv2.resize(best_prob, (w, h), interpolation=cv2.INTER_LINEAR)
                    hard_before = (best_prob > 0.5).astype(np.uint8)
                    hard = self._refine_mask(hard_before, fg2)
                    best_score = float(np.max(ious2))

        cand_tensor = torch.from_numpy(np.stack(cand_soft, axis=0).astype(np.float32)) if len(cand_soft) > 0 else None
        if self.vis_cb is not None:
            self.vis_cb("teacher_final", sample_id=sample_id, image=image, fg=fg, bg=bg, soft=best_prob, hard=hard, prm_before=hard_before, prm_after=hard, cand_soft=cand_soft)
        return PseudoPack(
            soft_mask=torch.from_numpy(best_prob.astype(np.float32)),
            hard_mask=torch.from_numpy(hard.astype(np.float32)),
            quality=torch.tensor(float(best_score), dtype=torch.float32),
            cand_soft_masks=cand_tensor,
            prm_before=torch.from_numpy(hard_before.astype(np.float32)),
            prm_after=torch.from_numpy(hard.astype(np.float32)),
        )
