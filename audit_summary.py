from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _safe_get(d: dict, keys: list[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_audit(audit_path: Path) -> dict:
    if not audit_path.exists():
        raise FileNotFoundError(f"Audit file not found: {audit_path}")
    with open(audit_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_summary_text(audit: dict, out_txt: Path):
    pseudo = audit.get("pseudo_refresh", [])
    phase = audit.get("phase_compare", [])
    epoch_eval = audit.get("epoch_eval", [])

    best = None
    if epoch_eval:
        best = max(epoch_eval, key=lambda x: _safe_get(x, ["overall", "mDice"], -1.0))

    lines = []
    lines.append("=== TU-SAM Audit Summary ===")
    lines.append(f"Pseudo refresh count: {len(pseudo)}")
    lines.append(f"Phase compare count: {len(phase)}")
    lines.append(f"Epoch eval count: {len(epoch_eval)}")

    if best is not None:
        lines.append(
            f"Best epoch: {best.get('epoch')} | mDice={_safe_get(best, ['overall', 'mDice'], 0.0):.6f} | "
            f"mIoU={_safe_get(best, ['overall', 'mIoU'], 0.0):.6f}"
        )

    if pseudo:
        lines.append("Pseudo refresh details:")
        for p in pseudo:
            lines.append(
                f"  - epoch={p.get('epoch')}, sigma_mean={p.get('sigma_mean', 0.0):.6f}, "
                f"sigma_std={p.get('sigma_std', 0.0):.6f}, hard_change_ratio={p.get('hard_change_ratio', 0.0):.6f}"
            )

    if phase:
        lines.append("Phase compare deltas:")
        for r in phase:
            lines.append(
                f"  - epoch={r.get('epoch')}, ΔmDice={_safe_get(r, ['delta', 'mDice'], 0.0):+.6f}, "
                f"ΔmIoU={_safe_get(r, ['delta', 'mIoU'], 0.0):+.6f}"
            )

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_curves(audit: dict, out_png: Path):
    pseudo = audit.get("pseudo_refresh", [])
    epoch_eval = audit.get("epoch_eval", [])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Overall mDice by epoch
    if epoch_eval:
        xs = [int(x.get("epoch", i + 1)) for i, x in enumerate(epoch_eval)]
        ys = [float(_safe_get(x, ["overall", "mDice"], 0.0)) for x in epoch_eval]
        axes[0].plot(xs, ys, marker="o", linewidth=1.8)
    axes[0].set_title("Overall mDice vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("mDice")
    axes[0].grid(True, alpha=0.3)

    # 2) sigma mean/std on pseudo refresh epochs
    if pseudo:
        ex = [int(x.get("epoch", i)) for i, x in enumerate(pseudo)]
        sm = [float(x.get("sigma_mean", 0.0)) for x in pseudo]
        ss = [float(x.get("sigma_std", 0.0)) for x in pseudo]
        axes[1].plot(ex, sm, marker="o", label="sigma_mean", linewidth=1.8)
        axes[1].plot(ex, ss, marker="s", label="sigma_std", linewidth=1.8)
        axes[1].legend()
    axes[1].set_title("Sigma Stats on Pseudo Refresh")
    axes[1].set_xlabel("Refresh Epoch")
    axes[1].set_ylabel("Value")
    axes[1].grid(True, alpha=0.3)

    # 3) hard change ratio on pseudo refresh epochs
    if pseudo:
        ex = [int(x.get("epoch", i)) for i, x in enumerate(pseudo)]
        hc = [float(x.get("hard_change_ratio", 0.0)) for x in pseudo]
        axes[2].plot(ex, hc, marker="^", color="tab:red", linewidth=1.8)
    axes[2].set_title("Hard Pseudo Change Ratio")
    axes[2].set_xlabel("Refresh Epoch")
    axes[2].set_ylabel("Ratio")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Summarize TU-SAM audit logs.")
    parser.add_argument(
        "--audit",
        type=str,
        default="outputs/tusam/audit.json",
        help="Path to audit.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/tusam",
        help="Directory to save summary outputs",
    )
    args = parser.parse_args()

    audit_path = Path(args.audit)
    out_dir = Path(args.out_dir)

    audit = load_audit(audit_path)
    txt_path = out_dir / "audit_summary.txt"
    png_path = out_dir / "audit_summary.png"

    save_summary_text(audit, txt_path)
    plot_curves(audit, png_path)

    print(f"Saved summary text: {txt_path}")
    print(f"Saved summary figure: {png_path}")


if __name__ == "__main__":
    main()
