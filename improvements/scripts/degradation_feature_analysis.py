"""退化特征可行性分析（EXP014，零训练）。

研究问题：能否仅凭 LR MRI 的频谱特征区分 Gaussian/Box/Fermi 退化？
比较三种 g 沿 through-plane（R，axis0）方向的归一化功率谱。

用法::

    python improvements/scripts/degradation_feature_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_mri


def power_spectrum_along_r(g: np.ndarray) -> np.ndarray:
    """沿 R（axis0）的一维平均功率谱（fftshift 顺序）。"""
    F = np.fft.fftshift(np.fft.fft(g, axis=0), axes=0)
    p = (np.abs(F) ** 2).mean(axis=(1, 2))
    return p / p.sum()  # 归一化


def main() -> int:
    s = "IXI002-Guys-0828-T1"; proc = Path("data/processed") / s
    # 沿 R 轴的频谱（归一化），比较三种退化
    print("归一化功率谱（沿 R 轴，前三/后三频率 bin 的相对能量）")
    for flt in ["gaussian", "box", "fermi"]:
        g_name = f"{s}_sinc_r2.nii.gz" if flt == "gaussian" else f"{s}_sinc_r2_{flt}.nii.gz"
        g = load_mri(proc / g_name, canonical=True).data.astype(np.float64)
        p = power_spectrum_along_r(g)
        n = len(p)
        low = p[:3].sum()  # 最低频
        high = p[-3:].sum()  # 最高频（fftshift 后两端）
        # 高频/低频比与谱峰
        print(f"{flt:<10} low-freq={low:.4f} high-freq={high:.6f} high/low={high/low:.5f}")

    # 两两相关性（不同退化之间的频谱相似度）
    spectra = {}
    for flt in ["gaussian", "box", "fermi"]:
        g_name = f"{s}_sinc_r2.nii.gz" if flt == "gaussian" else f"{s}_sinc_r2_{flt}.nii.gz"
        g = load_mri(proc / g_name, canonical=True).data.astype(np.float64)
        spectra[flt] = power_spectrum_along_r(g)
    print("\n频谱相关性（越高越难区分）:")
    for a in ["gaussian", "box", "fermi"]:
        for b in ["gaussian", "box", "fermi"]:
            if a < b:
                c = np.corrcoef(spectra[a], spectra[b])[0, 1]
                print(f"  {a} vs {b}: corr={c:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
