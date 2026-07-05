from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def tri_map(teacher_prob: torch.Tensor, tau_pos: float):
    tau_neg = 1.0 - tau_pos
    r_pos = (teacher_prob >= tau_pos).float()
    r_neg = (teacher_prob <= tau_neg).float()
    r_mid = 1.0 - r_pos - r_neg
    mv = r_pos + 0.5 * r_mid + 0.2 * r_neg
    return r_pos, r_mid, r_neg, mv


def edur_weight(teacher_prob: torch.Tensor, q: torch.Tensor, gamma: float = 2.0, return_parts: bool = False):
    eps = 1e-6
    h = -(teacher_prob * torch.log(teacher_prob + eps) + (1 - teacher_prob) * torch.log(1 - teacher_prob + eps))
    we = torch.pow(1 - h / math.log(2.0), gamma)
    wq = 0.5 + 0.5 * q.view(-1, 1, 1, 1)
    wc = (0.35 + 0.65 * we) * wq
    if return_parts:
        return wc, h, we, wq
    return wc


def masked_bce_dice(logits, target, weight, lambda_bce=0.6, lambda_dice=0.4):
    eps = 1e-6
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    lbce = torch.sum(weight * bce) / (torch.sum(weight) + eps)

    p = torch.sigmoid(logits)
    num = 2 * torch.sum(weight * p * target) + eps
    den = torch.sum(weight * p) + torch.sum(weight * target) + eps
    ldice = 1 - num / den
    return lambda_bce * lbce + lambda_dice * ldice, lbce.detach(), ldice.detach()


def boundary_map(prob: torch.Tensor, k: int = 5):
    pmax = F.max_pool2d(prob, kernel_size=k, stride=1, padding=k // 2)
    pmin = -F.max_pool2d(-prob, kernel_size=k, stride=1, padding=k // 2)
    return pmax - pmin


def _gaussian_kernel2d(kernel_size=7, sigma=1.0, device=None, dtype=None):
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2 + 1e-12))
    kernel = kernel / (kernel.sum() + 1e-12)
    return kernel


def _gaussian_blur(x: torch.Tensor, sigma=1.0, kernel_size=7):
    c = x.shape[1]
    k = _gaussian_kernel2d(kernel_size=kernel_size, sigma=sigma, device=x.device, dtype=x.dtype)
    k = k.view(1, 1, kernel_size, kernel_size).repeat(c, 1, 1, 1)
    return F.conv2d(x, k, padding=kernel_size // 2, groups=c)


def boundary_consistency(student_prob: torch.Tensor, teacher_prob: torch.Tensor, sigma_b: float = 1.0):
    eps = 1e-6
    es = boundary_map(student_prob)
    et = boundary_map(teacher_prob)
    et = _gaussian_blur(et, sigma=sigma_b, kernel_size=7)
    num = 2 * torch.sum(es * et) + eps
    den = torch.sum(es + et) + eps
    return 1 - num / den


def point_loss(student_prob: torch.Tensor, fg_points: torch.Tensor, bg_points: list[torch.Tensor]):
    eps = 1e-6
    b, _, h, w = student_prob.shape
    loss = 0.0
    for i in range(b):
        fx = int(torch.clamp(fg_points[i, 0], 0, w - 1).item())
        fy = int(torch.clamp(fg_points[i, 1], 0, h - 1).item())
        pf = torch.clamp(student_prob[i, 0, fy, fx], eps, 1 - eps)
        li = -torch.log(pf)

        bgs = bg_points[i]
        if bgs.numel() > 0:
            lbg = []
            for p in bgs:
                bx = int(torch.clamp(p[0], 0, w - 1).item())
                by = int(torch.clamp(p[1], 0, h - 1).item())
                pb = torch.clamp(student_prob[i, 0, by, bx], eps, 1 - eps)
                lbg.append(-torch.log(1 - pb))
            li = li + torch.stack(lbg).mean()
        loss = loss + li
    return loss / b


def bg_hinge_loss(gauss_map: torch.Tensor, bg_points: list[torch.Tensor], margin: float = 0.15):
    b, _, h, w = gauss_map.shape
    total = 0.0
    cnt = 0
    for i in range(b):
        bgs = bg_points[i]
        for p in bgs:
            x = int(torch.clamp(p[0], 0, w - 1).item())
            y = int(torch.clamp(p[1], 0, h - 1).item())
            total += torch.relu(gauss_map[i, 0, y, x] - margin)
            cnt += 1
    if cnt == 0:
        return gauss_map.sum() * 0.0
    return total / cnt


def weak_strong_consistency(weak_prob: torch.Tensor, strong_logits: torch.Tensor, r_mid: torch.Tensor, tau_c: float = 0.8, return_parts: bool = False):
    conf = torch.maximum(weak_prob, 1 - weak_prob)
    pseudo = (weak_prob > 0.5).float()
    mask = (conf > tau_c).float() * r_mid
    bce = F.binary_cross_entropy_with_logits(strong_logits, pseudo, reduction="none")
    den = mask.sum() + 1e-6
    loss_map = mask * bce
    loss = loss_map.sum() / den
    if return_parts:
        return loss, conf, mask, pseudo, loss_map
    return loss
