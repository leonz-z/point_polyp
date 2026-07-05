from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, "~/segment-anything")

from tusam.augment.mix_branch import foreground_aware_cutmix, foreground_center_crop, low_ratio_mixup, weak_geom
from tusam.data.dataset import PointTrainDataset, TestDataset
from tusam.losses.losses import (
    bg_hinge_loss,
    boundary_consistency,
    edur_weight,
    masked_bce_dice,
    point_loss,
    tri_map,
    weak_strong_consistency,
)
from tusam.models.student_factory import build_student, load_student_backbone_if_needed
from tusam.teacher.dgpa_teacher import FrozenSamTeacher
from tusam.utils.config import load_cfg


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_train(batch):
    ids = [x["id"] for x in batch]
    imgs = torch.stack([x["image"] for x in batch], dim=0)
    fgs = torch.stack([x["fg_point"] for x in batch], dim=0)
    bgs = [x["bg_points"] for x in batch]
    return {"id": ids, "image": imgs, "fg_point": fgs, "bg_points": bgs}






@torch.no_grad()
def precompute_pseudo(cfg, teacher, train_ds, epoch: int, prev_pseudo=None):
    pseudo = {}
    vis_enabled = bool(getattr(cfg, "pseudo_vis", False))
    vis_max = int(getattr(cfg, "pseudo_vis_max", 40))
    vis_unlimited = vis_max < 0
    exp_tag = str(getattr(cfg, "student_model", "pvt"))
    vis_dir = Path(getattr(cfg, "pseudo_vis_dir", str(Path(cfg.work_dir) / "pseudo_vis"))) / exp_tag / f"epoch_{epoch:03d}"
    if vis_enabled:
        vis_dir.mkdir(parents=True, exist_ok=True)

    vis_count = 0
    sigma_vals = []
    changed_ratio_sum = 0.0
    changed_count = 0
    for i in tqdm(range(len(train_ds)), desc=f"Pseudo@{epoch}"):
        item = train_ds[i]
        sid = item["id"]
        sigma = teacher.predict_sigma_single(item["image"], item["fg_point"])
        sigma_vals.append(float(sigma))
        p = teacher.generate_pseudo(item["image"], item["fg_point"], item["bg_points"], sigma=sigma, sample_id=sid)
        pseudo[sid] = p

        if prev_pseudo is not None and sid in prev_pseudo:
            prev_hard = (prev_pseudo[sid].hard_mask.numpy() > 0.5).astype(np.float32)
            cur_hard = (p.hard_mask.numpy() > 0.5).astype(np.float32)
            changed_ratio_sum += float(np.mean(np.abs(prev_hard - cur_hard)))
            changed_count += 1

        if vis_enabled and (vis_unlimited or vis_count < vis_max):
            base_name = f"{sid}"
            _save_pseudo_visualization(
                vis_dir / f"{base_name}.jpg",
                item["image"],
                item["fg_point"],
                item["bg_points"],
                sigma,
                p.soft_mask,
                p.hard_mask,
                p.cand_soft_masks,
                p.prm_before,
                p.prm_after,
            )
            _save_bw_mask(vis_dir / f"{base_name}_soft_bw.png", p.soft_mask.cpu().numpy())
            _save_bw_mask(vis_dir / f"{base_name}_hard_bw.png", p.hard_mask.cpu().numpy())
            if p.prm_before is not None:
                _save_bw_mask(vis_dir / f"{base_name}_prm_before_bw.png", p.prm_before.cpu().numpy())
            if p.prm_after is not None:
                _save_bw_mask(vis_dir / f"{base_name}_prm_after_bw.png", p.prm_after.cpu().numpy())
            vis_count += 1

    sigma_mean = float(np.mean(sigma_vals)) if len(sigma_vals) else 0.0
    sigma_std = float(np.std(sigma_vals)) if len(sigma_vals) else 0.0
    changed_ratio = float(changed_ratio_sum / max(1, changed_count))
    print(f"[PseudoAudit][epoch={epoch}] sigma_mean={sigma_mean:.5f}, sigma_std={sigma_std:.5f}, hard_change_ratio={changed_ratio:.5f}")

    pseudo_audit = {
        "epoch": int(epoch),
        "num_samples": int(len(train_ds)),
        "sigma_mean": sigma_mean,
        "sigma_std": sigma_std,
        "hard_change_ratio": changed_ratio,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    return pseudo, pseudo_audit


def tau_pos_schedule(epoch, total):
    r = min(1.0, epoch / max(1, total - 1))
    return 0.85 - 0.15 * r


def compute_metrics(pred, gt):
    pred = (pred > 0.5).float()
    gt = (gt > 0.5).float()
    inter = (pred * gt).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3))
    dice = (2 * inter + 1e-6) / (union + 1e-6)
    iou = (inter + 1e-6) / (pred.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3)) - inter + 1e-6)
    return dice.mean().item(), iou.mean().item()


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    flood = mask.copy().astype(np.uint8)
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 1)
    holes = 1 - flood
    return np.clip(mask + holes, 0, 1).astype(np.uint8)


def _post_process_single(prob_map: np.ndarray, threshold=0.5, keep_max_component=True, fill_holes=True) -> np.ndarray:
    m = (prob_map > threshold).astype(np.uint8)
    if keep_max_component:
        n, cc, st, _ = cv2.connectedComponentsWithStats(m, 8)
        if n > 1:
            best_id, best_score = 0, -1.0
            for k in range(1, n):
                comp = cc == k
                score = float(prob_map[comp].mean())
                if score > best_score:
                    best_score = score
                    best_id = k
            m = (cc == best_id).astype(np.uint8)
    if fill_holes:
        m = _fill_holes(m)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m.astype(np.float32)


def _pad_to_multiple(x: torch.Tensor, multiple: int = 32):
    h, w = x.shape[-2], x.shape[-1]
    nh = ((h + multiple - 1) // multiple) * multiple
    nw = ((w + multiple - 1) // multiple) * multiple
    ph = nh - h
    pw = nw - w
    if ph == 0 and pw == 0:
        return x, (0, 0, 0, 0)
    pt = ph // 2
    pb = ph - pt
    pl = pw // 2
    pr = pw - pl
    x = F.pad(x, (pl, pr, pt, pb), mode="reflect")
    return x, (pl, pr, pt, pb)


def _unpad(x: torch.Tensor, pads):
    pl, pr, pt, pb = pads
    if pl == 0 and pr == 0 and pt == 0 and pb == 0:
        return x
    h, w = x.shape[-2], x.shape[-1]
    return x[..., pt:h - pb, pl:w - pr]


@torch.no_grad()
def infer_tta(model, img, full=True, post_process=True, pp_cfg=None):
    views = [img, torch.flip(img, dims=[3]), torch.flip(img, dims=[2])]
    if full:
        views.append(F.interpolate(img, scale_factor=0.75, mode="bilinear", align_corners=False))
        views.append(F.interpolate(img, scale_factor=1.25, mode="bilinear", align_corners=False))

    probs = []
    for i, v in enumerate(views):
        v_pad, pads = _pad_to_multiple(v, multiple=32)
        logit, _, _ = model(v_pad)
        logit = _unpad(logit, pads)

        p = torch.sigmoid(logit)
        if v.shape[-2:] != img.shape[-2:]:
            p = F.interpolate(p, size=img.shape[-2:], mode="bilinear", align_corners=False)
        if i == 1:
            p = torch.flip(p, dims=[3])
        if i == 2:
            p = torch.flip(p, dims=[2])
        probs.append(p)

    p = torch.stack(probs, dim=0).mean(dim=0)
    if not post_process:
        return p

    pp_cfg = pp_cfg or {}
    out = []
    for i in range(p.shape[0]):
        pm = p[i, 0].detach().cpu().numpy()
        keep = _post_process_single(
            pm,
            threshold=float(pp_cfg.get("threshold", 0.5)),
            keep_max_component=bool(pp_cfg.get("keep_max_component", True)),
            fill_holes=bool(pp_cfg.get("fill_holes", True)),
        )
        out.append(torch.from_numpy(keep)[None])
    return torch.stack(out, dim=0).to(p.device)


@torch.no_grad()
def evaluate_all(cfg, model, device, full_tta=True):
    root = Path(cfg.test_root)
    results = {}
    for ds in sorted([d for d in os.listdir(root) if (root / d).is_dir()]):
        loader = DataLoader(TestDataset(str(root / ds), cfg.image_size), batch_size=8, shuffle=False, num_workers=4)
        dices, ious = [], []
        pp_cfg = {
            "threshold": float(getattr(cfg, "post_threshold", 0.5)),
            "keep_max_component": bool(getattr(cfg, "post_keep_max_component", True)),
            "fill_holes": bool(getattr(cfg, "post_fill_holes", True)),
        }
        for batch in loader:
            img = batch["image"].to(device)
            m = batch["mask"].to(device)
            p = infer_tta(model, img, full=full_tta, post_process=bool(getattr(cfg, "post_enable", True)), pp_cfg=pp_cfg)
            d, i = compute_metrics(p, m)
            dices.append(d)
            ious.append(i)
        results[ds] = {"mDice": float(np.mean(dices)), "mIoU": float(np.mean(ious))}
    results["Overall"] = {
        "mDice": float(np.mean([v["mDice"] for v in results.values()])),
        "mIoU": float(np.mean([v["mIoU"] for v in results.values()])),
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/tusam.yaml")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_ds = PointTrainDataset(cfg.train_root, cfg.train_points, cfg.image_size)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_train)

    student_name = str(getattr(cfg, "student_model", "pvt"))
    model = build_student(student_name).to(device)
    msg = load_student_backbone_if_needed(model, student_name, getattr(cfg, "pvt_ckpt", None))
    print(f"[Student] model={student_name}; {msg}")

    exp_tag = str(getattr(cfg, "student_model", "pvt"))
    work_dir = Path(cfg.work_dir) / exp_tag
    work_dir.mkdir(parents=True, exist_ok=True)

    teacher = FrozenSamTeacher(cfg.sam_type, cfg.sam_ckpt, device=str(device), sigma_min=cfg.sigma_min, sigma_max=cfg.sigma_max)

    params = list(model.parameters()) + list(teacher.dgpa.parameters())
    opt = AdamW(params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    sch = CosineAnnealingLR(opt, T_max=cfg.epochs)

    sigma_star_by_id = {sid: 0.08 for sid in train_ds.ids}

    audit_path = work_dir / "audit.json"
    audit = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config": str(args.config),
            "work_dir": str(work_dir),
            "epochs": int(cfg.epochs),
            "warmup_epochs": int(cfg.warmup_epochs),
            "pseudo_refresh_epochs": [int(x) for x in cfg.pseudo_refresh_epochs],
        },
        "pseudo_refresh": [],
        "phase_compare": [],
        "epoch_eval": [],
    }

    pseudo_cache, pseudo_audit = precompute_pseudo(cfg, teacher, train_ds, epoch=0, prev_pseudo=None)
    audit["pseudo_refresh"].append(pseudo_audit)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.eval_only:
        if (work_dir / "best.pth").exists():
            ck = torch.load(work_dir / "best.pth", map_location=device)
            if isinstance(ck, dict) and "student" in ck:
                model.load_state_dict(ck["student"], strict=True)
                if "dgpa" in ck:
                    teacher.dgpa.load_state_dict(ck["dgpa"], strict=False)
            else:
                model.load_state_dict(ck, strict=False)
        print(evaluate_all(cfg, model, device, full_tta=True))
        return

    best_overall = -1.0
    refresh_set = set(cfg.pseudo_refresh_epochs)

    for epoch in range(cfg.epochs):
        if epoch in refresh_set:
            model.eval()
            teacher.dgpa.eval()
            pre_res = evaluate_all(cfg, model, device, full_tta=True)
            print(f"[PhaseCompare][epoch={epoch}] before_refresh: {pre_res['Overall']}")

            old_cache = pseudo_cache
            pseudo_cache, pseudo_audit = precompute_pseudo(cfg, teacher, train_ds, epoch=epoch, prev_pseudo=old_cache)
            audit["pseudo_refresh"].append(pseudo_audit)

            post_res = evaluate_all(cfg, model, device, full_tta=True)
            print(f"[PhaseCompare][epoch={epoch}] after_refresh: {post_res['Overall']}")

            audit["phase_compare"].append(
                {
                    "epoch": int(epoch),
                    "before": {
                        "mDice": float(pre_res["Overall"]["mDice"]),
                        "mIoU": float(pre_res["Overall"]["mIoU"]),
                    },
                    "after": {
                        "mDice": float(post_res["Overall"]["mDice"]),
                        "mIoU": float(post_res["Overall"]["mIoU"]),
                    },
                    "delta": {
                        "mDice": float(post_res["Overall"]["mDice"] - pre_res["Overall"]["mDice"]),
                        "mIoU": float(post_res["Overall"]["mIoU"] - pre_res["Overall"]["mIoU"]),
                    },
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            )
            audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
            model.train()
            teacher.dgpa.train()

        model.train()
        teacher.dgpa.train()
        tau_pos = tau_pos_schedule(epoch, cfg.epochs)
        mix_w = 0.0 if epoch < cfg.warmup_epochs else cfg.lambda_mix_max * (epoch - cfg.warmup_epochs + 1) / (cfg.epochs - cfg.warmup_epochs)

        train_vis_enabled = False
        train_vis_count = 0
        aug_counter = Counter()
        aug_values = defaultdict(list)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for batch in pbar:
            img = batch["image"].to(device)
            fg = batch["fg_point"].to(device)
            bg_list = [x.to(device) for x in batch["bg_points"]]
            b, _, h, w = img.shape

            t_soft = torch.stack([pseudo_cache[s].soft_mask for s in batch["id"]], dim=0).to(device)[:, None]
            q = torch.stack([pseudo_cache[s].quality for s in batch["id"]], dim=0).to(device)

            r_pos, r_mid, r_neg, mv = tri_map(t_soft, tau_pos=tau_pos)
            wc, h_pt, we, wq = edur_weight(t_soft, q, return_parts=True)
            wb = 1 + 5 * torch.abs(F.avg_pool2d(t_soft, kernel_size=5, stride=1, padding=2) - t_soft)
            w_all = wb * mv * wc

            logits, aux1, aux2 = model(img)
            prob = torch.sigmoid(logits)

            l_sup, _, _ = masked_bce_dice(logits, t_soft, w_all, cfg.lambda_bce, cfg.lambda_dice)
            l_bd = boundary_consistency(prob, t_soft, sigma_b=1.0)
            l_pt = point_loss(prob, fg, bg_list)

            sigma_pred = teacher.predict_sigma_batch(img, fg)
            if epoch < cfg.warmup_epochs:
                l_sigma = sigma_pred.mean() * 0.0
            else:
                pbin = (prob.detach() > 0.5).float()
                area = pbin.flatten(1).mean(dim=1)
                valid = (area > 0.001) & (area < 0.4)
                target_sigma = []
                for i, sid in enumerate(batch["id"]):
                    if valid[i]:
                        rs = float(torch.sqrt(area[i] / np.pi + 1e-6).item())
                        sigma_star_by_id[sid] = 0.9 * sigma_star_by_id[sid] + 0.1 * rs
                    target_sigma.append(sigma_star_by_id[sid])
                target_sigma = torch.tensor(target_sigma, device=device)
                l_sigma = cfg.sigma_alpha * torch.mean(torch.abs(sigma_pred - target_sigma))

            ys = torch.arange(h, device=device).view(1, h, 1).repeat(b, 1, w)
            xs = torch.arange(w, device=device).view(1, 1, w).repeat(b, h, 1)
            sig_pix = (sigma_pred.detach().view(b, 1, 1) * max(h, w)).clamp(min=1e-6)
            gmap = torch.exp(-((xs - fg[:, 0].view(b, 1, 1)) ** 2 + (ys - fg[:, 1].view(b, 1, 1)) ** 2) / (2 * sig_pix**2 + 1e-6))[:, None]
            l_bg = bg_hinge_loss(gmap, bg_list, margin=0.15)

            weak_prob = prob.detach()
            strong_img = torch.clamp(img + torch.empty_like(img).uniform_(-0.15, 0.15), 0.0, 1.0)
            strong_logits, _, _ = model(strong_img)
            l_con, conf_w, con_mask, con_pseudo, con_loss_map = weak_strong_consistency(weak_prob, strong_logits, r_mid, tau_c=0.8, return_parts=True)

            l_aux1, _, _ = masked_bce_dice(aux1, t_soft, w_all, cfg.lambda_bce, cfg.lambda_dice)
            l_aux2, _, _ = masked_bce_dice(aux2, t_soft, w_all, cfg.lambda_bce, cfg.lambda_dice)
            l_orig = l_sup + cfg.lambda_bd * l_bd + cfg.lambda_sigma * l_sigma + cfg.lambda_bg * l_bg + cfg.lambda_pt * l_pt + cfg.lambda_con * l_con

            l_mix = logits.sum() * 0.0
            if epoch >= cfg.warmup_epochs:
                mix_imgs, mix_masks = [], []
                crop_samples = []
                cutmix_samples = []
                perm = torch.randperm(b)
                if b > 1:
                    idx = torch.arange(b, device=perm.device)
                    for _ in range(16):
                        if not torch.any(perm == idx):
                            break
                        perm = torch.randperm(b, device=perm.device)
                    if torch.any(perm == idx):
                        perm = torch.roll(idx, shifts=1)
                for i in range(b):
                    mi, mm = foreground_center_crop(img[i], t_soft[i, 0], fg_xy=fg[i], scales=tuple(cfg.crop_scales))
                    crop_samples.append((batch["id"][i], img[i].detach().cpu(), mi.detach().cpu(), t_soft[i, 0].detach().cpu(), mm.detach().cpu()))
                    mi, mm = weak_geom(mi, mm)
                    j = perm[i].item()
                    dj, dm = img[j], t_soft[j, 0]
                    # record host (after crop+geom) and donor before cutmix
                    host_img = mi.detach().cpu().clone()
                    host_mask = mm.detach().cpu().clone()
                    donor_img = dj.detach().cpu().clone()
                    donor_mask = dm.detach().cpu().clone()
                    mi, mm, cutmix_info = foreground_aware_cutmix(mi, mm, dj, dm, pmax=cfg.cutmix_pmax, tau_upper=cfg.cutmix_tau_upper, return_info=True)
                    # record before mixup (after cutmix)
                    before_mixup_img = mi.detach().cpu().clone()
                    before_mixup_mask = mm.detach().cpu().clone()
                    donor_mix_img = dj.detach().cpu().clone()
                    donor_mix_mask = dm.detach().cpu().clone()
                    mi, mm, mixup_info = low_ratio_mixup(mi, mm, dj, dm, cfg.mixup_lambda_range[0], cfg.mixup_lambda_range[1], cfg.mixup_prob, return_info=True)
                    # record after mixup (final)
                    after_mixup_img = mi.detach().cpu().clone()
                    after_mixup_mask = mm.detach().cpu().clone()
                    cutmix_samples.append((batch["id"][i], host_img, host_mask, donor_img, donor_mask, before_mixup_img, before_mixup_mask, after_mixup_img, after_mixup_mask, cutmix_info, mixup_info, donor_mix_img, donor_mix_mask))
                    aug_counter[f"cutmix_{cutmix_info['status']}"] += 1
                    aug_values["cutmix_host_ratio"].append(float(cutmix_info.get("host_ratio", 0.0)))
                    aug_values["cutmix_prob"].append(float(cutmix_info.get("prob", 0.0)))
                    mix_imgs.append(mi)
                    mix_masks.append(mm)
                    aug_counter[f"mixup_{mixup_info['status']}"] += 1
                    aug_values["mixup_lambda"].append(float(mixup_info.get("lambda", 1.0)))
                mix_imgs = torch.stack(mix_imgs, dim=0).to(device)
                mix_masks = torch.stack(mix_masks, dim=0).to(device)[:, None]
                mix_logits, _, _ = model(mix_imgs)
                l_mix, _, _ = masked_bce_dice(mix_logits, mix_masks, torch.ones_like(mix_masks), cfg.lambda_bce, cfg.lambda_dice)


            loss = l_orig + mix_w * l_mix + 0.5 * l_aux1 + 0.3 * l_aux2

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


            pbar.set_postfix(loss=float(loss.detach().cpu()), lsup=float(l_sup.detach().cpu()), lmix=float(l_mix.detach().cpu()), lsig=float(l_sigma.detach().cpu()))

        sch.step()
        torch.save({"student": model.state_dict(), "dgpa": teacher.dgpa.state_dict()}, work_dir / "last.pth")

        model.eval()
        teacher.dgpa.eval()
        res = evaluate_all(cfg, model, device, full_tta=True)
        print(f"Epoch {epoch+1} results: {res}")

        total_cut = sum(v for k, v in aug_counter.items() if k.startswith("cutmix_"))
        total_mix = sum(v for k, v in aug_counter.items() if k.startswith("mixup_"))
        aug_audit = {
            "epoch": int(epoch + 1),
            "cutmix": {
                "total": int(total_cut),
                "success": int(aug_counter.get("cutmix_success", 0)),
                "success_rate": float(aug_counter.get("cutmix_success", 0) / max(1, total_cut)),
                "stats": {k: int(v) for k, v in aug_counter.items() if k.startswith("cutmix_")},
                "host_ratio_mean": float(np.mean(aug_values["cutmix_host_ratio"])) if aug_values["cutmix_host_ratio"] else 0.0,
                "prob_mean": float(np.mean(aug_values["cutmix_prob"])) if aug_values["cutmix_prob"] else 0.0,
            },
            "mixup": {
                "total": int(total_mix),
                "success": int(aug_counter.get("mixup_success", 0)),
                "success_rate": float(aug_counter.get("mixup_success", 0) / max(1, total_mix)),
                "stats": {k: int(v) for k, v in aug_counter.items() if k.startswith("mixup_")},
                "lambda_mean": float(np.mean(aug_values["mixup_lambda"])) if aug_values["mixup_lambda"] else 1.0,
            },
        }
        audit.setdefault("aug_stats", []).append(aug_audit)
        audit["epoch_eval"].append(
            {
                "epoch": int(epoch + 1),
                "overall": {
                    "mDice": float(res["Overall"]["mDice"]),
                    "mIoU": float(res["Overall"]["mIoU"]),
                },
                "datasets": {k: {"mDice": float(v["mDice"]), "mIoU": float(v["mIoU"])} for k, v in res.items() if k != "Overall"},
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

        if res["Overall"]["mDice"] > best_overall:
            best_overall = res["Overall"]["mDice"]
            torch.save({"student": model.state_dict(), "dgpa": teacher.dgpa.state_dict()}, work_dir / "best.pth")

    print("Training done. Best Overall mDice:", best_overall)


if __name__ == "__main__":
    main()
