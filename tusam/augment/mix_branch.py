from __future__ import annotations

import random

import cv2
import numpy as np
import torch


def foreground_center_crop(image: torch.Tensor, mask: torch.Tensor, fg_xy: torch.Tensor | None = None, scales=(0.5, 0.75, 1.0), weights=None, min_size=64):
    # Prefer the annotated foreground point as crop center to match the paper.
    c, h, w = image.shape
    if weights is None:
        weights = [0.4, 0.35, 0.25] if len(scales) == 3 else None

    if fg_xy is not None:
        s = random.choices(scales, weights=weights)[0]
        cx = int(torch.clamp(fg_xy[0], 0, w - 1).item())
        cy = int(torch.clamp(fg_xy[1], 0, h - 1).item())
    else:
        ys, xs = torch.where(mask > 0.5)
        if len(xs) == 0:
            s = 1.0
            cx, cy = w // 2, h // 2
        else:
            s = random.choices(scales, weights=weights)[0]
            cx = int(xs.float().mean().item())
            cy = int(ys.float().mean().item())

    ch, cw = max(min_size, int(h * s)), max(min_size, int(w * s))
    ch, cw = min(ch, h), min(cw, w)
    x1 = max(0, min(w - cw, cx - cw // 2))
    y1 = max(0, min(h - ch, cy - ch // 2))
    x2, y2 = x1 + cw, y1 + ch

    img = image[:, y1:y2, x1:x2]
    m = mask[y1:y2, x1:x2]

    img = torch.nn.functional.interpolate(img[None], size=(h, w), mode="bilinear", align_corners=False)[0]
    m = torch.nn.functional.interpolate(m[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0].clamp(0.0, 1.0)
    return img, m


def _bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def foreground_aware_cutmix(host_img, host_mask, donor_img, donor_mask, pmax=0.45, tau_upper=0.10, eta=0.15, max_trials=30, max_overlap=0.02, min_fg_area=20, max_patch_ratio=0.45, return_info=False):
    h, w = host_mask.shape
    r = float((host_mask > 0.5).float().mean().item())
    p = pmax * max(0.0, 1.0 - r / tau_upper)
    info = {
        "status": "init",
        "applied": False,
        "prob": float(p),
        "host_ratio": float(r),
        "src_bbox": None,
        "dst_bbox": None,
        "changed_pixels": 0,
    }
    if random.random() > p:
        info["status"] = "skip_prob"
        return (host_img, host_mask, info) if return_info else (host_img, host_mask)

    fg = donor_mask > 0.5
    ys, xs = torch.where(fg)
    if len(xs) == 0:
        info["status"] = "skip_invalid_donor"
        return (host_img, host_mask, info) if return_info else (host_img, host_mask)

    x1, x2 = int(xs.min().item()), int(xs.max().item()) + 1
    y1, y2 = int(ys.min().item()), int(ys.max().item()) + 1
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * eta))
    y1 = max(0, int(y1 - bh * eta))
    x2 = min(w, int(x2 + bw * eta))
    y2 = min(h, int(y2 + bh * eta))
    info["src_bbox"] = [int(x1), int(y1), int(x2), int(y2)]

    pw, ph = x2 - x1, y2 - y1
    if pw <= 1 or ph <= 1:
        info["status"] = "skip_invalid_bbox"
        return (host_img, host_mask, info) if return_info else (host_img, host_mask)

    patch_ratio = (pw * ph) / float(h * w)
    src_m = donor_mask[y1:y2, x1:x2].clamp(0.0, 1.0)
    donor_fg_area = int((src_m > 0.5).sum().item())
    if donor_fg_area < min_fg_area:
        info["status"] = "skip_tiny_donor"
        return (host_img, host_mask, info) if return_info else (host_img, host_mask)
    if patch_ratio > max_patch_ratio:
        info["status"] = "skip_huge_patch"
        return (host_img, host_mask, info) if return_info else (host_img, host_mask)

    src_img = donor_img[:, y1:y2, x1:x2]
    src_fg = src_m > 0.5
    dst = None
    for _ in range(max_trials):
        dx1 = random.randint(0, max(0, w - pw))
        dy1 = random.randint(0, max(0, h - ph))
        dx2, dy2 = dx1 + pw, dy1 + ph
        host_region = host_mask[dy1:dy2, dx1:dx2]
        overlap = ((host_region > 0.5) & src_fg).float().mean().item()
        if overlap <= max_overlap:
            dst = (dx1, dy1, dx2, dy2)
            break

    if dst is None:
        info["status"] = "skip_no_valid_dst"
        return (host_img, host_mask, info) if return_info else (host_img, host_mask)

    dx1, dy1, dx2, dy2 = dst
    out_img = host_img.clone()
    out_m = host_mask.clone()
    alpha = src_m[None].clamp(0.0, 1.0)
    out_img[:, dy1:dy2, dx1:dx2] = alpha * src_img + (1.0 - alpha) * out_img[:, dy1:dy2, dx1:dx2]
    out_m[dy1:dy2, dx1:dx2] = torch.maximum(out_m[dy1:dy2, dx1:dx2], src_m)
    changed_pixels = int((torch.mean(torch.abs(out_img - host_img), dim=0) > (5.0 / 255.0)).sum().item())
    info["changed_pixels"] = changed_pixels
    if changed_pixels <= 100:
        info["status"] = "skip_no_change"
        return (host_img, host_mask, info) if return_info else (host_img, host_mask)

    info.update({"status": "success", "applied": True, "dst_bbox": [int(dx1), int(dy1), int(dx2), int(dy2)]})
    return (out_img, out_m, info) if return_info else (out_img, out_m)


def low_ratio_mixup(img1, m1, img2, m2, lam_low=0.65, lam_high=0.85, p=0.15, return_info=False):
    info = {"status": "success", "applied": False, "lambda": 1.0, "changed_pixels": 0}
    if random.random() > p:
        info["status"] = "skip_prob"
        return (img1, m1, info) if return_info else (img1, m1)
    lam = random.uniform(lam_low, lam_high)
    out_img = lam * img1 + (1 - lam) * img2
    out_m = lam * m1 + (1 - lam) * m2
    changed_pixels = int((torch.mean(torch.abs(out_img - img1), dim=0) > (3.0 / 255.0)).sum().item())
    info.update({"applied": changed_pixels > 100, "lambda": float(lam), "changed_pixels": changed_pixels})
    if not info["applied"]:
        info["status"] = "skip_no_change"
        return (img1, m1, info) if return_info else (img1, m1)
    return (out_img, out_m, info) if return_info else (out_img, out_m)


def weak_geom(image: torch.Tensor, mask: torch.Tensor):
    if random.random() < 0.3:
        image = torch.flip(image, dims=[2])
        mask = torch.flip(mask, dims=[1])
    if random.random() < 0.2:
        image = torch.flip(image, dims=[1])
        mask = torch.flip(mask, dims=[0])
    if random.random() < 0.2:
        dev_i, dev_m = image.device, mask.device
        dt_i, dt_m = image.dtype, mask.dtype
        ang = random.uniform(-15, 15)
        mat = cv2.getRotationMatrix2D((image.shape[2] // 2, image.shape[1] // 2), ang, 1.0)
        i = image.permute(1, 2, 0).detach().cpu().numpy()
        m = mask.detach().cpu().numpy()
        i = cv2.warpAffine(i, mat, (image.shape[2], image.shape[1]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        m = cv2.warpAffine(m, mat, (image.shape[2], image.shape[1]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        image = torch.from_numpy(i).permute(2, 0, 1).to(device=dev_i, dtype=dt_i)
        mask = torch.from_numpy(m).to(device=dev_m, dtype=dt_m).clamp(0.0, 1.0)
    return image, mask
