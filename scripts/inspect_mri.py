"""检查 NIfTI MRI 文件并生成 axial/coronal/sagittal 三视图。

用法（在项目根目录运行）::

    python scripts/inspect_mri.py                          # 扫描 data/raw 下全部 NIfTI
    python scripts/inspect_mri.py path/to/IXI002-T1.nii.gz
    python scripts/inspect_mri.py --data-dir data/raw/ixi --no-figures

输出：
- 终端打印每例的 shape / voxel spacing / orientation / 强度统计；
- results/mri_inspection.json 保存完整结构化检查结果；
- results/figures/<文件名>_orthogonal.png 保存三视图。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许直接运行 `python scripts/inspect_mri.py` 时导入 src 包
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")  # 无窗口后端，保证 PNG 一定写盘
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import (
    SUPPORTED_SUFFIXES,
    inspect_mri,
    load_mri,
    normalize_mri,
    orthogonal_slices,
)


def find_nifti_files(root: Path) -> list[Path]:
    """递归查找 root 下的 .nii / .nii.gz 文件。"""
    if not root.exists():
        return []
    files = [p for p in root.rglob("*") if p.is_file() and p.name.lower().endswith(SUPPORTED_SUFFIXES)]
    return sorted(files)


def _display_volume(volume) -> np.ndarray:
    """把原始强度映射到适合灰度显示的 [0, 1]，基于前景分位数。"""
    return normalize_mri(
        volume.data,
        method="percentile",
        lower_percentile=0.5,
        upper_percentile=99.5,
        background_threshold=0.0,
        clip=True,
    )


def save_orthogonal_figure(vol, out_path: Path, *, title: str | None = None) -> Path:
    """保存同一受试者的三个正交中心切片（canonical RAS+ 方向）。

    使用以毫米为单位的 extent + aspect='equal'，因此不同体素间距下
    图像的长宽比例仍然是真实物理比例，不会被拉伸失真。
    """
    display = _display_volume(vol)
    views = orthogonal_slices(display, vol.voxel_spacing)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), constrained_layout=True)
    for ax, (name, info) in zip(axes, views.items()):
        ax.imshow(
            info["slice"],
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            origin="lower",
            extent=info["extent"],
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_title(f"{name.capitalize()}  (slice {info['index']} along {info['index_axis']})")
        ax.set_xlabel(f"{info['x_axis']}  [mm]", fontsize=8)
        ax.set_ylabel(f"{info['y_axis']}  [mm]", fontsize=8)
        ax.tick_params(labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("0.7")

    spacing = " x ".join(f"{s:g}" for s in vol.voxel_spacing)
    figure_title = title or vol.path.name
    fig.suptitle(
        f"{figure_title}\n"
        f"canonical RAS+ shape = {vol.shape}, voxel spacing = {spacing} mm",
        fontsize=11,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    return out_path


def print_report(report: dict) -> None:
    """把检查结果以中文打印到终端。"""
    inten = report["intensity"]
    levels = inten["percentile_levels"]
    p_all = inten["percentiles_all"]
    p_fg = inten["percentiles_foreground"]
    src = report.get("canonical_ras", {})

    def _fmt(values):
        return ", ".join(f"P{int(q)}={v:.2f}" for q, v in zip(levels, values))

    print("=" * 78)
    print(f"文件: {report['file_name']}")
    print(f"  路径            : {report['path']}")
    print(f"  大小            : {report['file_size_bytes'] / 1024**2:.2f} MB")
    print(f"  Shape           : {tuple(report['shape'])}  (dtype={report['dtype']})")
    print(f"  Voxel spacing   : {tuple(report['voxel_spacing_mm'])} mm (来自 NIfTI header)")
    print(f"  Orientation     : {report['orientation']}  (affine 方向码)")
    print(f"  Affine Matrix   : (voxel -> world, mm)")
    for row in report["affine"]:
        print("      [" + ", ".join(f"{v:9.4f}" for v in row) + "]")
    print(f"  Affine 行列式   : {report['affine_determinant']:.4f}")
    print(f"  强度 min/max    : {inten['min']:.2f} / {inten['max']:.2f}")
    print(f"  全体分位数      : {_fmt(p_all)}")
    print(f"  前景分位数      : {_fmt(p_fg)}  (foreground={inten['foreground_voxels']}, "
          f"{100 * inten['foreground_fraction']:.2f}%)")
    print(f"  NaN / Inf       : {report['has_nan']} / {report['has_inf']} "
          f"(nonfinite voxels={report['nonfinite_voxels']})")
    print(f"  有效三维图像    : {report['is_valid_3d']}")
    if src:
        print(f"  canonical RAS+  : shape={tuple(src['shape'])}, "
              f"spacing={tuple(src['voxel_spacing_mm'])} mm, orientation={src['orientation']}")
        if src["shape_changed"] or src["orientation_changed"]:
            print("    (注意: canonical 重排改变了轴顺序/方向，原始文件未被修改)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 IXI T1 MRI 数据并生成三视图")
    parser.add_argument("paths", nargs="*", type=Path, help="NIfTI 文件路径（可选）")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="未提供 paths 时递归扫描该目录（默认 data/raw）",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "figures",
        help="三视图输出目录（默认 results/figures）",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "results" / "mri_inspection.json",
        help="结构化检查结果 JSON 路径",
    )
    parser.add_argument("--no-figures", action="store_true", help="只检查，不生成图片")
    parser.add_argument("--no-canonical", action="store_true", help="不计算 canonical RAS+ 信息")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    files = list(args.paths) if args.paths else find_nifti_files(args.data_dir)

    if not files:
        print(f"未找到任何 NIfTI 文件。请将 .nii / .nii.gz 放入: {args.data_dir}")
        return 1

    reports: list[dict] = []
    failures: list[tuple[str, str]] = []

    for path in files:
        try:
            report = inspect_mri(path, compute_canonical=not args.no_canonical)
            print_report(report)
            reports.append(report)

            if not args.no_figures:
                vol = load_mri(path, canonical=True)
                out = args.figures_dir / f"{vol.path.name.replace('.nii.gz', '').replace('.nii', '')}_orthogonal.png"
                save_orthogonal_figure(vol, out)
                print(f"  三视图已保存    : {out}")
        except Exception as exc:  # 单个文件失败不影响其余文件
            failures.append((str(path), repr(exc)))
            print("-" * 78)
            print(f"[失败] {path}: {exc}")

    if reports:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_files": len(reports),
            "files": reports,
            "failures": [{"path": p, "error": e} for p, e in failures],
        }
        args.output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("=" * 78)
        print(f"检查结果 JSON 已保存: {args.output_json}")

    print(f"\n汇总: 成功 {len(reports)} 例, 失败 {len(failures)} 例, 共 {len(files)} 个文件。")
    return 0 if reports and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
