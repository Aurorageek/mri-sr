"""生成论文级图表（Phase 3）。

Figures:
- fig3_mismatch_heatmap: 开发受试者 3x4 ΔPSNR 热力图（EXP012）
- fig4_robustness_independent: 独立受试者 Fixed-G vs Fixed-Box 稳健性（EXP015）
- fig5_reconstruction: 独立受试者 HR/Sinc/Fixed-G/Fixed-Box + 误差图

用法::

    python improvements/scripts/make_paper_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import load_mri
from src.edsr import EDSR
from src.metrics import brain_mask
from src.metrics import psnr

FIG_DIR = PROJECT_ROOT / "improvements" / "figures"
METHODS = ["Fixed-G", "Fixed-Box"]


def fig3_heatmap():
    # EXP012 数据（IXI002，ΔPSNR vs Sinc）
    data = np.array([
        [0.273, 0.222],   # gaussian actual
        [-1.637, 0.340],  # box actual
        [-1.236, 0.291],  # fermi actual
    ])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Fixed-Gaussian", "Fixed-Box"], fontsize=9)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["Gaussian", "Box", "Fermi"], fontsize=9)
    ax.set_xlabel("Training degradation (self-supervised)")
    ax.set_ylabel("Actual degradation (input)")
    for i in range(3):
        for j in range(2):
            ax.text(j, i, f"{data[i, j]:+.2f}", ha="center", va="center", fontsize=11,
                    color="black")
    ax.set_title("ΔPSNR vs Sinc (dB) — degradation mismatch", fontsize=10)
    fig.colorbar(im, ax=ax, label="ΔPSNR (dB)")
    fig.tight_layout()
    out = FIG_DIR / "paper" / "fig3_mismatch_heatmap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor="white"); plt.close(fig)
    print("saved", out)


def fig4_robustness():
    subjects = ["IXI022", "IXI023", "IXI024", "IXI025"]
    g = {"gauss": [0.243, 0.240, 0.238, 0.321], "box": [-1.611, -1.678, -1.663, -1.811],
         "fermi": [-1.194, -1.313, -1.251, -1.365]}
    b = {"gauss": [0.062, 0.220, 0.221, 0.180], "box": [0.167, 0.258, 0.370, 0.291],
         "fermi": [0.286, 0.338, 0.290, 0.238]}
    x = np.arange(len(subjects)); w = 0.26
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, (cond, color) in enumerate(zip(["gauss", "box", "fermi"], ["#3498db", "#e67e22", "#8e44ad"])):
        ax.bar(x + (k - 1) * w, g[cond], w * 0.45, color=color, alpha=0.45, label=f"Fixed-G ({cond})")
        ax.bar(x + (k - 1) * w + w * 0.45, b[cond], w * 0.45, color=color, label=f"Fixed-Box ({cond})")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(subjects)
    ax.set_ylabel("ΔPSNR vs Sinc (dB)")
    ax.set_title("Independent subjects: Fixed-Gaussian (light) vs Fixed-Box (solid)")
    ax.legend(fontsize=7, ncol=3, loc="lower left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "paper" / "fig4_robustness_independent.png"
    fig.savefig(out, dpi=160, facecolor="white"); plt.close(fig)
    print("saved", out)


def load_model(p, dev):
    ck = torch.load(p, map_location=dev); m = ck.get("config", {}).get("model", {})
    model = EDSR(m.get("n_resblocks", 8), m.get("n_feats", 64), m.get("res_scale", 0.1)).to(dev)
    model.load_state_dict(ck["model_state_dict"]); model.eval(); return model


def infer(model, g, dev, bs=4):
    n_r, n_a, n_s = g.shape; out = np.zeros_like(g, dtype=np.float32)
    with torch.no_grad():
        for s0 in range(0, n_s, bs):
            s1 = min(s0 + bs, n_s)
            batch = np.concatenate([g[:, :, s][None, None] for s in range(s0, s1)], 0)
            y = model(torch.from_numpy(batch).float().to(dev)).cpu().numpy()
            for k, s in enumerate(range(s0, s1)):
                out[:, :, s] = y[k, 0]
    return out


def fig5_reconstruction():
    dev = torch.device("cuda")
    s = "IXI022-Guys-0701-T1"; proc = PROJECT_ROOT / "data" / "processed" / s
    hr_vol = load_mri(proc / f"{s}_HR.nii.gz", canonical=True)
    hr = hr_vol.data.astype(np.float64)
    g_vol = load_mri(proc / f"{s}_sinc_r2.nii.gz", canonical=True)
    g = g_vol.data.astype(np.float64)
    scale = float(np.percentile(g[g > 0], 99.5)) if (g > 0).any() else 1.0
    models = {
        "Fixed-G": load_model(PROJECT_ROOT / "results/checkpoints" / f"{s}_indep_fixedg" / "best.pt", dev),
        "Fixed-Box": load_model(PROJECT_ROOT / "results/checkpoints" / f"{s}_indep_fixedbox" / "best.pt", dev),
    }
    recs = {"Sinc": g}
    for name, model in models.items():
        recs[name] = infer(model, (g / scale).astype(np.float32), dev) * scale

    # 统一显示窗口（HR 前景分位数）
    fg = hr[hr > 0]
    lo, hi = np.percentile(fg, [0.5, 99.5])

    def win(x):
        return np.clip((x - lo) / (hi - lo), 0, 1)

    # Coronal（fix A=1）与 Axial（fix S=2）——均含 R 方向
    views = {"coronal": 1, "axial": 2}
    names = ["HR", "Sinc", "Fixed-G", "Fixed-Box"]
    for view, cut in views.items():
        center = hr.shape[cut] // 2
        fig, axes = plt.subplots(len(names) + 1, 2, figsize=(7, 3.2 * (len(names) + 1)))
        for r, name in enumerate(names):
            arr = hr if name == "HR" else recs[name]
            slc = np.take(arr, center, axis=cut).T
            axes[r, 0].imshow(win(slc), cmap="gray", vmin=0, vmax=1, origin="lower", aspect="equal")
            axes[r, 0].set_title(name, fontsize=8); axes[r, 0].axis("off")
            # zoom
            h, w = slc.shape
            z0, z1 = int(h * 0.3), int(h * 0.7); w0, w1 = int(w * 0.3), int(w * 0.7)
            axes[r, 1].imshow(win(slc)[z0:z1, w0:w1], cmap="gray", vmin=0, vmax=1,
                              origin="lower", aspect="equal")
            axes[r, 1].set_title(f"{name} (zoom)", fontsize=8); axes[r, 1].axis("off")
        # 误差行
        for j, name in enumerate(["Fixed-G", "Fixed-Box"]):
            err = np.abs(win(hr) - win(recs[name]))
            slc = np.take(err, center, axis=cut).T
            axes[-1, j].imshow(slc, cmap="hot", vmin=0, vmax=0.3, origin="lower", aspect="equal")
            axes[-1, j].set_title(f"|HR-{name}|", fontsize=8); axes[-1, j].axis("off")
        fig.suptitle(f"{s} — {view} (Gaussian degradation, r=2)", fontsize=10)
        fig.tight_layout()
        out = FIG_DIR / "paper" / f"fig5_reconstruction_{s}_{view}.png"
        fig.savefig(out, dpi=150, facecolor="white"); plt.close(fig)
        print("saved", out)


def main() -> int:
    fig3_heatmap()
    fig4_robustness()
    fig5_reconstruction()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
