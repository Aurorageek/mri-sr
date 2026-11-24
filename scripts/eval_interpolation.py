"""Day 2 端到端：through-plane 退化 + Sinc 插值 baseline + 指标 + 可视化。

对 data/raw/ixi 下每例 T1：

1. canonical RAS+ 读取 HR；
2. 沿 through-plane 轴（axis 0 = R）做 r=2 抗混叠退化 -> LR（粗网格）；
3. k-space 零填充 + IFFT -> Sinc 插值结果（HR 网格）；
4. 脑组织 mask 内计算 PSNR / SSIM；
5. 保存 HR / LR / Sinc 插值为 NIfTI；
6. 保存 HR-vs-Sinc-vs-|error| 图（Coronal + Sagittal）与 k-space 频谱图。

用法::

    python scripts/eval_interpolation.py
    python scripts/eval_interpolation.py --r 2 --filter gaussian
    python scripts/eval_interpolation.py --subjects IXI002-Guys-0828-T1.nii.gz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from src.dataset import SUPPORTED_SUFFIXES, load_mri, normalize_mri
from src.degradation import (
    degrade_through_plane,
    kspace_power_spectrum,
    sinc_interpolate,
)
from src.metrics import brain_mask, evaluate

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ixi"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "results" / "figures"

AXIS_NAMES = ("R", "A", "S")


def find_nifti(subjects: list[str] | None = None) -> list[Path]:
    if subjects:
        files = [RAW_DIR / s for s in subjects]
    else:
        files = sorted(
            p for p in RAW_DIR.glob("*") if p.is_file() and p.name.lower().endswith(SUPPORTED_SUFFIXES)
        )
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"找不到数据文件: {missing}")
    return files


def _display(volume: np.ndarray) -> np.ndarray:
    """前景分位数归一化到 [0,1]，仅用于可视化。"""
    return normalize_mri(volume, method="percentile", lower_percentile=0.5, upper_percentile=99.5)


def save_volume(data: np.ndarray, affine: np.ndarray, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(img, str(out))
    return out


def _slice_at(arr: np.ndarray, axis: int, index: int) -> np.ndarray:
    sl = [slice(None)] * 3
    sl[axis] = index
    return arr[tuple(sl)]


def save_comparison_figure(
    hr: np.ndarray,
    sinc: np.ndarray,
    spacing: tuple,
    axis: int,
    subj_stem: str,
    out_dir: Path,
) -> list[Path]:
    """同一位置 HR / Sinc / |误差| 三行，列=Coronal、Sagittal（含退化轴）。"""
    out_paths = []
    out_dir.mkdir(parents=True, exist_ok=True)
    # canonical 轴: 0=R, 1=A, 2=S。退化轴 R，包含 R 的两个正交面是 coronal(R-A) 与 sagittal(R-S)
    views = {
        "coronal": (1, {"x_axis": "R", "y_axis": "S"}),   # 切面垂直于 A，包含 R
        "sagittal": (2, {"x_axis": "A", "y_axis": "S"}),  # 切面垂直于 R，包含 R
    }
    hr_d = _display(hr)
    sinc_d = _display(sinc)
    err = np.abs(hr_d - sinc_d)

    for view_name, (cut_axis, _) in views.items():
        center = hr.shape[cut_axis] // 2
        rows = []
        for arr, label in [(hr_d, "HR"), (sinc_d, "Sinc"), (err, "|Error|")]:
            rows.append((arr, label))
        fig, axes = plt.subplots(3, 1, figsize=(6, 15), constrained_layout=True)
        for ax, (arr, label) in zip(axes, rows):
            slc = _slice_at(arr, cut_axis, center).T
            im = ax.imshow(slc, cmap="gray", origin="lower", aspect="equal",
                           interpolation="nearest")
            ax.set_title(f"{label}  ({view_name} view, cut={AXIS_NAMES[cut_axis]}={center})")
            ax.set_xlabel(AXIS_NAMES[axis] + " [voxel]")
            if label == "|Error|":
                fig.colorbar(im, ax=ax, fraction=0.046)
        spc = " x ".join(f"{s:g}" for s in spacing)
        fig.suptitle(f"{subj_stem}\nthrough-plane degradation (r=2) + Sinc baseline — "
                     f"{view_name} view\nHR spacing = {spc} mm", fontsize=10)
        out = out_dir / f"{subj_stem}_{view_name}_hr_sinc_error.png"
        fig.savefig(out, dpi=140, facecolor="white")
        plt.close(fig)
        out_paths.append(out)
    return out_paths


def save_spectrum_figure(
    hr: np.ndarray,
    lr: np.ndarray,
    sinc: np.ndarray,
    axis: int,
    r: float,
    subj_stem: str,
    out_dir: Path,
) -> Path:
    """沿退化轴的一维平均功率谱：HR vs 截断后（LR 零填充前）。"""
    f_hr = np.fft.fftshift(np.fft.fftfreq(hr.shape[axis]))
    p_hr = kspace_power_spectrum(hr, axis)
    # LR 在粗网格上的谱，再零填充到 HR 网格对应频率（用于同轴比较）
    f_lr = np.fft.fftshift(np.fft.fftfreq(lr.shape[axis]))
    p_lr = kspace_power_spectrum(lr, axis)
    # Sinc 插值后的谱（应集中在 |f| <= 0.5/r 内）
    p_sinc = kspace_power_spectrum(sinc, axis)
    f_sinc = f_hr

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(f_hr, p_hr / p_hr.max(), label="HR", lw=1.2)
    ax.semilogy(f_sinc, p_sinc / p_sinc.max(), label="Sinc-interp (band-limited)", lw=1.2, alpha=0.8)
    # 用 LR 谱在粗频率上标注截断点
    f_cut = 0.5 / r
    ax.axvline(f_cut, color="red", ls="--", lw=1, label=f"f_cut = 0.5/{r:g}")
    ax.axvline(-f_cut, color="red", ls="--", lw=1)
    ax.set_xlabel(f"spatial frequency along {AXIS_NAMES[axis]} [cycles/voxel]")
    ax.set_ylabel("normalized mean power |FFT|^2")
    ax.set_title(f"{subj_stem}: through-plane ({AXIS_NAMES[axis]}) k-space before/after truncation")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    out = out_dir / f"{subj_stem}_kspace_spectrum.png"
    fig.savefig(out, dpi=140, facecolor="white")
    plt.close(fig)
    return out


def process_subject(path: Path, axis: int, r: float, filter_name: str) -> dict:
    vol = load_mri(path, canonical=True)
    hr = vol.data.astype(np.float64)
    stem = vol.path.name.replace(".nii.gz", "").replace(".nii", "")

    deg = degrade_through_plane(
        hr, axis=axis, r=r, spacing=vol.voxel_spacing, affine=vol.affine, filter_name=filter_name
    )
    sinc = sinc_interpolate(deg.lr, axis=axis, r=r, target_n=hr.shape[axis])

    # 保存 NIfTI
    subj_dir = PROC_DIR / stem
    hr_out = save_volume(hr, vol.affine, subj_dir / f"{stem}_HR.nii.gz")
    lr_out = save_volume(deg.lr, deg.lr_affine, subj_dir / f"{stem}_LR_r{int(r)}.nii.gz")
    sinc_out = save_volume(sinc, vol.affine, subj_dir / f"{stem}_sinc_r{int(r)}.nii.gz")

    # 指标
    mask = brain_mask(hr)
    met = evaluate(hr, sinc, mask)

    # 可视化
    fig_out = FIG_DIR / "degradation"
    figs = save_comparison_figure(hr, sinc, vol.voxel_spacing, axis, stem, fig_out)
    spec_fig = save_spectrum_figure(hr, deg.lr, sinc, axis, r, stem, fig_out)

    return {
        "subject": stem,
        "raw_path": str(vol.path),
        "canonical_shape": list(hr.shape),
        "canonical_spacing_mm": [round(v, 6) for v in vol.voxel_spacing],
        "axis": axis,
        "axis_name": AXIS_NAMES[axis],
        "r": r,
        "filter": filter_name,
        "lr_shape": list(deg.lr.shape),
        "lr_spacing_mm": [round(v, 6) for v in deg.lr_spacing],
        "imag_residual_max": float(np.max(np.abs(np.imag(
            np.fft.ifft(np.fft.fft(deg.lr, axis=axis), axis=axis)
        )))),
        "metrics": {k: (v if not isinstance(v, float) or np.isfinite(v) else None) for k, v in met.items()},
        "outputs": {
            "hr": str(hr_out.relative_to(PROJECT_ROOT)),
            "lr": str(lr_out.relative_to(PROJECT_ROOT)),
            "sinc": str(sinc_out.relative_to(PROJECT_ROOT)),
            "figures": [str(f.relative_to(PROJECT_ROOT)) for f in figs + [spec_fig]],
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Day 2: 退化 + Sinc 插值 baseline 评估")
    p.add_argument("--r", type=float, default=2.0)
    p.add_argument("--axis", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--filter", default="gaussian", choices=["gaussian", "box", "fermi"])
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "results" / "day2_metrics.json")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    files = find_nifti(args.subjects)
    results = []
    for path in files:
        print(f"处理 {path.name} (axis={AXIS_NAMES[args.axis]}, r={args.r}, filter={args.filter})")
        try:
            rec = process_subject(path, args.axis, args.r, args.filter)
            results.append(rec)
            m = rec["metrics"]
            print(f"  LR shape={rec['lr_shape']}  spacing={rec['lr_spacing_mm']} mm")
            print(f"  PSNR={m['psnr_db']:.3f} dB  SSIM={m['ssim']:.5f}  MSE={m['mse']:.4f}  "
                  f"mask={m['mask_voxels']} voxels")
        except Exception as exc:  # 单例失败不阻断其余
            print(f"  [失败] {type(exc).__name__}: {exc}")

    payload = {
        "task": "Day2 through-plane degradation + Sinc interpolation baseline",
        "params": {"axis": args.axis, "axis_name": AXIS_NAMES[args.axis], "r": args.r,
                   "filter": args.filter},
        "subjects": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n指标 JSON 已保存: {args.output_json}")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
