"""方向 A：退化算子对齐的数值分析（EXP001，无需训练）。

研究问题：SMORE 自监督训练时的退化（对 in-plane 切片做单轴高斯低通）与推理时
实际输入 g 经历的退化（高斯低通 + 下采样 + Sinc 插值）是否等价？

本脚本用 1D 合成信号与真实 MRI 切片比较两条退化路径：
- D_train：gaussian_lowpass(x)  （训练输入 b）
- D_real ：sinc_interpolate(downsample(gaussian_lowpass(x)))  （实际输入 g）

输出两者的频域谱、MSE/相关系数，判断是否存在值得研究的失配。

用法::

    python improvements/scripts/analyze_degradation_alignment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_mri
from src.degradation import degrade_through_plane, lowpass_along_axis, sinc_interpolate


def d_train(x: np.ndarray, r: float, filter_name: str = "gaussian") -> np.ndarray:
    """训练退化：单轴低通（不下采样、不插值）。"""
    return lowpass_along_axis(x, axis=0, r=r, filter_name=filter_name)


def d_real(x: np.ndarray, r: float, filter_name: str = "gaussian") -> np.ndarray:
    """实际退化：低通 + 下采样 + Sinc 插值，恢复到原网格。"""
    n = x.shape[0]
    deg = degrade_through_plane(x[..., None, None], axis=0, r=r, filter_name=filter_name)
    lr = deg.lr[..., 0, 0]
    return sinc_interpolate(lr[..., None, None], axis=0, r=r, target_n=n)[..., 0, 0]


def spectrum(x: np.ndarray) -> np.ndarray:
    F = np.fft.fftshift(np.fft.fft(x))
    return F


def compare(name: str, x: np.ndarray, r: float = 2.0) -> dict:
    b = d_train(x, r)
    g = d_real(x, r)
    f = np.fft.fftshift(np.fft.fftfreq(len(x)))
    sp_x = np.abs(spectrum(x)) ** 2
    sp_b = np.abs(spectrum(b)) ** 2
    sp_g = np.abs(spectrum(g)) ** 2
    mse_bg = float(np.mean((b - g) ** 2))
    corr = float(np.corrcoef(b, g)[0, 1])
    # 高频段（|f|>0.25）能量占比
    hi = np.abs(f) > 0.25
    frac_b_hi = float(sp_b[hi].sum() / sp_b.sum())
    frac_g_hi = float(sp_g[hi].sum() / sp_g.sum())
    print(f"[{name}] len={len(x)}")
    print(f"  D_train 高频(>0.25)能量占比 = {frac_b_hi:.4f}   D_real = {frac_g_hi:.4f}")
    print(f"  MSE(D_train, D_real) = {mse_bg:.6f}   corr = {corr:.6f}")
    return {"mse": mse_bg, "corr": corr, "frac_b_hi": frac_b_hi, "frac_g_hi": frac_g_hi}


def _band_real(n: int, fmax: float, rng: np.random.Generator) -> np.ndarray:
    """生成严格带限实信号（Hermitian 频谱，DC/Nyquist=0）。"""
    F = np.zeros(n, dtype=np.complex128)
    for k in range(1, n // 2):
        if k / n < fmax:
            F[k] = rng.standard_normal() + 1j * rng.standard_normal()
            F[n - k] = np.conj(F[k])
    return np.fft.ifft(F).real


def main() -> int:
    print("===== 1. 合成信号：带限低频 =====")
    rng = np.random.default_rng(0)
    n = 256
    x_band = _band_real(n, 0.15, rng)
    compare("band-limited (f<0.15)", x_band)

    print("\n===== 2. 合成信号：含高频（f 到 0.45） =====")
    x_high = _band_real(n, 0.45, rng)
    compare("high-freq (f<0.45)", x_high)

    print("\n===== 3. 真实 MRI 切片（沿 R 轴 1D 剖面） =====")
    vol = load_mri(Path("data/processed/IXI002-Guys-0828-T1/IXI002-Guys-0828-T1_HR.nii.gz"),
                   canonical=True)
    hr = vol.data.astype(np.float64)
    # 取若干条沿 R（axis0）的 1D 剖面
    r = 2.0
    profs = [hr[:, a, s] for a, s in [(128, 128), (100, 150), (150, 100)]]
    for i, p in enumerate(profs):
        compare(f"real R-profile {i}", p, r)

    print("\n结论：若 D_real 高频占比明显低于 D_train 且两者 MSE 大，则训练退化与实际退化不等价。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
