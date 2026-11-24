"""Day 5：MRI 结构保真分析可视化。

同一受试者、同一空间位置，统一灰度窗口，比较 HR / Sinc / SMORE-L1 / SMORE-Grad：
- 切片对比（含局部放大）
- 绝对误差图
- 梯度误差图（|Sobel(HR) - Sobel(method)|，作为结构保真代理）

输出到 results/figures/error_maps/。

用法::

    python scripts/analyze_structure.py --subject IXI013-HH-1212-T1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

from src.dataset import load_mri


def _window(hr, lo=0.5, hi=99.5):
    fg = hr[hr > 0]
    return float(np.percentile(fg, lo)), float(np.percentile(fg, hi))


def _apply(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _slice(arr, cut, idx):
    sl = [slice(None)] * 3; sl[cut] = idx
    return arr[tuple(sl)]


def _zoom(slc, frac=0.5):
    h, w = slc.shape
    z0, z1 = int(h * (0.5 - frac / 2)), int(h * (0.5 + frac / 2))
    w0, w1 = int(w * (0.5 - frac / 2)), int(w * (0.5 + frac / 2))
    return slc[z0:z1, w0:w1]


def _show(ax, slc, title, cmap="gray", vmin=0, vmax=1):
    ax.imshow(slc.T, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower",
              aspect="equal", interpolation="nearest")
    ax.set_title(title, fontsize=8); ax.axis("off")


def _grad_mag(x):
    return ndimage.sobel(x)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Day 5 结构保真分析")
    p.add_argument("--subject", default="IXI013-HH-1212-T1")
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "figures" / "error_maps")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    proc = PROJECT_ROOT / "data" / "processed" / args.subject
    hr = load_mri(proc / f"{args.subject}_HR.nii.gz", canonical=True).data.astype(np.float64)
    methods = {
        "Sinc": load_mri(proc / f"{args.subject}_sinc_r2.nii.gz", canonical=True).data.astype(np.float64),
        "SMORE-L1": load_mri(proc / f"{args.subject}_smore_l1.nii.gz", canonical=True).data.astype(np.float64),
        "SMORE-Grad": load_mri(proc / f"{args.subject}_smore_grad.nii.gz", canonical=True).data.astype(np.float64),
    }

    lo, hi = _window(hr)
    hr_d = _apply(hr, lo, hi)
    disp = {k: _apply(v, lo, hi) for k, v in methods.items()}

    views = {"coronal": 1, "sagittal": 2}
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for view, cut in views.items():
        center = hr.shape[cut] // 2
        # 切片对比（含放大）
        names = ["HR"] + list(disp.keys())
        fig, axes = plt.subplots(len(names), 2, figsize=(9, 3.2 * len(names)), constrained_layout=True)
        for r, name in enumerate(names):
            arr = hr_d if name == "HR" else disp[name]
            slc = _slice(arr, cut, center)
            _show(axes[r, 0], slc, f"{name} ({view})")
            _show(axes[r, 1], _zoom(slc), f"{name} (zoom)")
        fig.suptitle(f"{args.subject}: {view} slices (window [{lo:.0f},{hi:.0f}])", fontsize=10)
        fig.savefig(out_dir / f"{args.subject}_{view}_slices.png", dpi=140, facecolor="white"); plt.close(fig)

        # 绝对误差图
        fig, axes = plt.subplots(len(disp), 2, figsize=(9, 3.2 * len(disp)), constrained_layout=True)
        for r, name in enumerate(disp.keys()):
            err = np.abs(hr_d - disp[name])
            slc = _slice(err, cut, center)
            _show(axes[r, 0], slc, f"|HR-{name}|", cmap="hot", vmax=0.4)
            _show(axes[r, 1], _zoom(slc), f"|HR-{name}| (zoom)", cmap="hot", vmax=0.4)
        fig.suptitle(f"{args.subject}: {view} absolute error", fontsize=10)
        fig.savefig(out_dir / f"{args.subject}_{view}_abs_error.png", dpi=140, facecolor="white"); plt.close(fig)

        # 梯度误差图（结构保真代理）
        hr_g = _grad_mag(hr_d)
        fig, axes = plt.subplots(len(disp), 2, figsize=(9, 3.2 * len(disp)), constrained_layout=True)
        for r, name in enumerate(disp.keys()):
            err = np.abs(hr_g - _grad_mag(disp[name]))
            slc = _slice(err, cut, center)
            _show(axes[r, 0], slc, f"|grad HR - grad {name}|", cmap="hot", vmax=0.15)
            _show(axes[r, 1], _zoom(slc), f"grad err (zoom)", cmap="hot", vmax=0.15)
        fig.suptitle(f"{args.subject}: {view} gradient error (structure fidelity proxy)", fontsize=10)
        fig.savefig(out_dir / f"{args.subject}_{view}_grad_error.png", dpi=140, facecolor="white"); plt.close(fig)

    print(f"结构保真图已保存到 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
