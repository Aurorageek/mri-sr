"""Day 3/4：多方法对比评价（Sinc / SMORE-L1 / SMORE-L1+Gradient）。

统一评价协议（与 Day 2/3 一致）：
- HR 只在最终评价读取；
- 同一 brain mask、同一原始强度域、同一 PSNR data_range；
- 指标：PSNR / SSIM / MAE / 边缘误差（Sobel 梯度幅值 MAE）。

用法::

    python scripts/eval_smore.py --methods sinc smore_day4_l1 smore_day4_grad010
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import load_mri
from src.metrics import brain_mask, evaluate_with_edge


def _window(hr: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> tuple[float, float]:
    fg = hr[hr > 0]
    return float(np.percentile(fg, lo)), float(np.percentile(fg, hi))


def _apply_window(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _slice(arr: np.ndarray, cut_axis: int, idx: int) -> np.ndarray:
    sl = [slice(None)] * 3
    sl[cut_axis] = idx
    return arr[tuple(sl)]


def _slice2d(arr: np.ndarray, cut_axis: int, idx: int) -> np.ndarray:
    return _slice(arr, cut_axis, idx)


def _zoom(slc2d: np.ndarray, frac: float = 0.45) -> np.ndarray:
    """裁出中心 frac 比例区域。"""
    h, w = slc2d.shape
    z0, z1 = int(h * (0.5 - frac / 2)), int(h * (0.5 + frac / 2))
    w0, w1 = int(w * (0.5 - frac / 2)), int(w * (0.5 + frac / 2))
    return slc2d[z0:z1, w0:w1]


def _show(ax, slc2d, title, cmap="gray", vmin=0.0, vmax=1.0):
    ax.imshow(slc2d.T, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower",
              aspect="equal", interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def save_comparison_figure(
    hr, images: dict[str, np.ndarray], subject: str, out_dir: Path,
    axis_names=("R", "A", "S")
) -> list[Path]:
    """Coronal/Sagittal：统一窗口切片图（含局部放大）与绝对误差图。"""
    lo, hi = _window(hr)
    hr_d = _apply_window(hr, lo, hi)
    disp = {k: _apply_window(v, lo, hi) for k, v in images.items()}

    views = {
        # canonical RAS+：fix A(axis1) -> R-S 平面 = coronal；fix S(axis2) -> R-A 平面 = axial
        "coronal": (1, axis_names[0] + "-" + axis_names[2]),
        "axial": (2, axis_names[0] + "-" + axis_names[1]),
    }
    out_paths = []
    names = ["HR"] + list(disp.keys())
    for view_name, (cut_axis, plane_label) in views.items():
        center = hr.shape[cut_axis] // 2

        # 图 1：切片 + 局部放大（左列全图，右列放大）
        fig, axes = plt.subplots(len(names), 2, figsize=(9, 3.2 * len(names)),
                                 constrained_layout=True)
        for r, name in enumerate(names):
            arr = hr_d if name == "HR" else disp[name]
            slc = _slice2d(arr, cut_axis, center)
            _show(axes[r, 0], slc, f"{name} ({view_name})")
            _show(axes[r, 1], _zoom(slc), f"{name} (zoom)")
        fig.suptitle(f"{subject}: {view_name} view (unified window [{lo:.0f},{hi:.0f}])", fontsize=10)
        out = out_dir / f"{subject}_{view_name}_slices.png"
        fig.savefig(out, dpi=140, facecolor="white"); plt.close(fig)
        out_paths.append(out)

        # 图 2：绝对误差图（每方法一行，左全图右放大）
        fig, axes = plt.subplots(len(disp), 2, figsize=(9, 3.2 * len(disp)),
                                 constrained_layout=True)
        for r, name in enumerate(disp.keys()):
            err = np.abs(hr_d - disp[name])
            slc = _slice2d(err, cut_axis, center)
            _show(axes[r, 0], slc, f"|HR - {name}|", cmap="hot", vmax=0.4)
            _show(axes[r, 1], _zoom(slc), f"|HR - {name}| (zoom)", cmap="hot", vmax=0.4)
        fig.suptitle(f"{subject}: {view_name} absolute error (unified window)", fontsize=10)
        out = out_dir / f"{subject}_{view_name}_errors.png"
        fig.savefig(out, dpi=140, facecolor="white"); plt.close(fig)
        out_paths.append(out)
    return out_paths


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Day 4 多方法评价")
    p.add_argument("--subject", default="IXI002-Guys-0828-T1")
    p.add_argument("--methods", nargs="+",
                   default=["sinc", "smore_day4_l1", "smore_day4_grad001",
                            "smore_day4_grad005", "smore_day4_grad010"],
                   help="sinc 或 smore_<name>，对应 data/processed/<subject>/<subject>_<name>.nii.gz")
    p.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "results" / "day4_metrics.json")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    proc = PROJECT_ROOT / "data" / "processed" / args.subject
    hr_path = proc / f"{args.subject}_HR.nii.gz"

    hr = load_mri(hr_path, canonical=True).data.astype(np.float64)
    mask = brain_mask(hr)

    images = {}; metrics = {}
    for m in args.methods:
        if m == "sinc":
            p = proc / f"{args.subject}_sinc_r2.nii.gz"
            label = "Sinc"
        else:
            p = proc / f"{args.subject}_{m}.nii.gz"
            label = m
        if not p.exists():
            print(f"[跳过] 缺失 {p}")
            continue
        vol = load_mri(p, canonical=True).data.astype(np.float64)
        if vol.shape != hr.shape:
            print(f"[跳过] {m} 形状 {vol.shape} != HR {hr.shape}")
            continue
        images[label] = vol
        metrics[label] = evaluate_with_edge(hr, vol, mask)

    fig_dir = PROJECT_ROOT / "results" / "figures" / "comparison"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figs = save_comparison_figure(hr, images, args.subject, fig_dir)

    # 打印对比表
    print(f"\n{'Method':<24} {'PSNR':>8} {'SSIM':>8} {'MAE':>10} {'GradMAE':>10}")
    print("-" * 64)
    for label, m in metrics.items():
        print(f"{label:<24} {m['psnr_db']:>8.3f} {m['ssim']:>8.5f} "
              f"{m['mae']:>10.3f} {m['gradient_mae']:>10.4f}")

    result = {
        "subject": args.subject,
        "mask_voxels": int(np.count_nonzero(mask)),
        "methods": {k: {kk: (vv if not isinstance(vv, float) or np.isfinite(vv) else None)
                        for kk, vv in v.items()} for k, v in metrics.items()},
        "figures": [str(f.relative_to(PROJECT_ROOT)) for f in figs],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果 JSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
