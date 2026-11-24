"""各向异性 MRI through-plane 退化与 Sinc 插值（Day 2）。

对应 SMORE（IEEE TMI, 2021）的采集模型
========================================

论文 Section II-A 把 MRI 采样写成 ``y = S x + n``，其中 ``S`` 包含
(1) 层选激励（slice-selective excitation，近似一个沿层方向的低通轮廓）与
(2) 沿 through-plane 方向的稀疏采样（层间距大于层内分辨率）。

Section III-A 的模拟退化即：
    HR -> 沿 through-plane 轴做抗混叠低通 -> 按因子 r 下采样 -> LR（粗网格）。

本模块把这一过程分解为两个可复用算子：

* :func:`degrade_through_plane`：低通 + 下采样，得到**真实低采样网格**上的 LR。
* :func:`sinc_interpolate`：对 LR 做 k-space 零填充 + IFFT，得到**数字恢复网格**
  （与 HR 同尺寸）上的 Sinc 插值结果。

关键区分
--------

* **真实低采样网格**：LR 在该方向只有 ``ceil(n/r)`` 个样本，体素间距为 HR 的 r 倍。
* **数字恢复网格**：Sinc 插值把 LR 恢复到 HR 的 n 个采样位置，但**并不能凭空恢复**
  被低通/截断丢掉的高频信息；它只是理想带限内插，等价于对 HR 施加理想矩形低通。

低通滤波器
----------

沿 through-plane 轴在频域（k-space）施加，对应论文中"层选轮廓 / 抗混叠"：

* ``"gaussian"``（默认，SMORE 采集模型）：H(f) = exp(-ln2·(f/f_c)²)，
  半功率点在 f_c = Nyquist/r，模拟层选激励的平滑抗混叠轮廓。
* ``"box"``：理想矩形截断 H(f)=1{|f|<=f_c}。它正好等于 Sinc 插值的核，
  用于演示"若采集与重建都假设理想带限"这一退化解（会产生 Gibbs 振铃）。
* ``"fermi"``：平滑过渡窗，抗振铃（对应论文标题里的 anti-aliasing 思想）。

所有运算都正确处理 ``fftshift`` / ``ifftshift`` 与频域索引；逆 FFT 后取实部，
虚部残差在数值精度量级（见 tests）。

注意：NIfTI 里存的是**幅度图**（magnitude），不是复数 k-space；对幅度图做 FFT
截断是一种**模拟**，不等于真实少采相位编码线。这一点在 README 中明确说明。

本阶段不做重采样到各向同性网格（除非显式指定），以避免引入额外插值误差。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

import numpy as np

SUPPORTED_FILTERS = ("gaussian", "box", "fermi")


def _centered_freq(n: int, spacing: float = 1.0) -> np.ndarray:
    """返回 fftshift 顺序的离散频率轴（cycles / voxel），DC 居中。

    例：n=150 时从 -0.5 到 +0.4966...（含唯一 Nyquist 点 -0.5）。
    """
    return np.fft.fftshift(np.fft.fftfreq(n, d=spacing))


def _lowpass_mask(n: int, r: float, filter_name: str) -> np.ndarray:
    """沿单轴的一维低通 mask（fftshift 顺序，长度 n）。

    Parameters
    ----------
    n : int
        该轴的体素数。
    r : float
        退化因子。f_cut = 0.5 / r cycles/voxel（Nyquist 除以 r）。
    filter_name : {"gaussian", "box", "fermi"}
    """
    if filter_name not in SUPPORTED_FILTERS:
        raise ValueError(f"未知滤波器 {filter_name!r}，可选 {SUPPORTED_FILTERS}")
    if r < 1.0:
        raise ValueError(f"r 必须 >= 1，得到 {r}")
    f = _centered_freq(n)
    f_cut = 0.5 / r
    if filter_name == "box":
        return (np.abs(f) <= f_cut).astype(np.float64)
    if filter_name == "gaussian":
        # 半功率点位于 f_cut
        return np.exp(-np.log(2.0) * (f / f_cut) ** 2)
    # fermi
    width = max(0.5 / n, 0.01)  # 过渡带宽与网格分辨率同量级
    return 1.0 / (1.0 + np.exp((np.abs(f) - f_cut) / width))


def _broadcast_mask(mask: np.ndarray, axis: int, shape: Tuple[int, ...]) -> np.ndarray:
    """把一维 mask 广播成与数组同形（任意维度），非退化轴为 1。"""
    shp = [1] * len(shape)
    shp[axis] = len(mask)
    return mask.reshape(shp).astype(np.float64)


@dataclass
class DegradedVolume:
    """一次 through-plane 退化的结果。"""

    lr: np.ndarray              # 真实低采样网格上的 LR（float64）
    filtered: np.ndarray        # 低通后的 HR 网格体数据（float64，用于调试/对照）
    axis: int
    r: float
    filter_name: str
    lr_spacing: Tuple[float, float, float]
    lr_affine: np.ndarray
    hr_shape: Tuple[int, ...]
    params: Dict[str, float] = field(default_factory=dict)


def degrade_through_plane(
    volume: np.ndarray,
    axis: int,
    r: float,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    affine: np.ndarray | None = None,
    filter_name: str = "gaussian",
) -> DegradedVolume:
    """沿指定轴做抗混叠低通并下采样 r 倍，模拟厚层采集。

    Parameters
    ----------
    volume : np.ndarray
        三维 HR 体数据（canonical RAS+）。不会被修改。
    axis : int
        退化轴（0=R, 1=A, 2=S）。对于本项目的矢状位 IXI 数据，through-plane
        低分辨率轴是 R，即 axis=0。
    r : float
        退化因子（>=1）。r=1 表示恒等（不下采样、不滤波）。
    spacing : Sequence[float]
        HR 三个轴的体素间距（mm），用于更新 LR 间距。
    affine : np.ndarray | None
        HR 的 4x4 affine；提供时返回更新后的 LR affine。
    filter_name : {"gaussian", "box", "fermi"}
        低通滤波器类型。

    Returns
    -------
    DegradedVolume
    """
    vol = np.asarray(volume, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError(f"degrade_through_plane 期望三维数组，得到 {vol.shape}")
    if not (0 <= axis < 3):
        raise ValueError(f"axis 必须是 0/1/2，得到 {axis}")
    if r < 1.0:
        raise ValueError(f"r 必须 >= 1，得到 {r}")

    n = vol.shape[axis]
    spacing = tuple(float(s) for s in spacing)
    if len(spacing) != 3 or any(s <= 0 for s in spacing):
        raise ValueError(f"非法 spacing: {spacing}")

    if r == 1:
        return DegradedVolume(
            lr=vol.copy(),
            filtered=vol.copy(),
            axis=axis,
            r=1.0,
            filter_name="identity",
            lr_spacing=spacing,
            lr_affine=np.array(affine, dtype=float) if affine is not None else np.eye(4),
            hr_shape=vol.shape,
            params={"r": 1.0, "f_cut": 0.5, "filter": "identity"},
        )

    # 1) 沿 axis 做 FFT，频域乘低通 mask，再 IFFT
    mask = _lowpass_mask(n, r, filter_name)
    F = np.fft.fft(vol, axis=axis)
    F = np.fft.fftshift(F, axes=axis)
    F = F * _broadcast_mask(mask, axis, vol.shape)
    F = np.fft.ifftshift(F, axes=axis)
    filtered = np.fft.ifft(F, axis=axis).real  # HR 网格上的低通结果

    # 2) 沿 axis 下采样 r 倍 -> 真实低采样网格
    sl = [slice(None)] * 3
    sl[axis] = slice(0, n, int(round(r))) if float(r).is_integer() else slice(0, n)
    lr = filtered[tuple(sl)]

    # 3) 更新体素间距与 affine
    lr_spacing = list(spacing)
    lr_spacing[axis] = lr_spacing[axis] * r
    lr_affine = None
    if affine is not None:
        A = np.array(affine, dtype=float).copy()
        A[:, axis] = A[:, axis] * r  # 方向向量缩放 r，平移不变（voxel 0 对齐）
        lr_affine = A

    return DegradedVolume(
        lr=lr,
        filtered=filtered,
        axis=axis,
        r=float(r),
        filter_name=filter_name,
        lr_spacing=tuple(lr_spacing),
        lr_affine=lr_affine,
        hr_shape=vol.shape,
        params={
            "r": float(r),
            "f_cut": 0.5 / r,
            "filter": filter_name,
            "lr_length": int(lr.shape[axis]),
            "hr_length": int(n),
        },
    )


def sinc_interpolate(
    lr: np.ndarray,
    axis: int,
    r: float,
    target_n: int | None = None,
) -> np.ndarray:
    """对 LR 做 k-space 零填充 + 逆 FFT，得到 Sinc 插值结果。

    这是"传统插值 baseline"：把粗网格上的 LR 恢复回 HR 网格，只做理想带限内插，
    不学习、不重建丢失的高频。

    Parameters
    ----------
    lr : np.ndarray
        粗网格 LR（float64）。
    axis : int
        插值轴。
    r : float
        上采样因子（对应退化因子）。
    target_n : int | None
        目标长度；默认 ``lr.shape[axis] * r``。

    Returns
    -------
    np.ndarray
        与 HR 网格对齐的 Sinc 插值体数据（float64，与 lr 除 axis 外同形）。
    """
    vol = np.asarray(lr, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError(f"sinc_interpolate 期望三维数组，得到 {vol.shape}")
    if not (0 <= axis < 3):
        raise ValueError(f"axis 必须是 0/1/2，得到 {axis}")
    if r < 1.0:
        raise ValueError(f"r 必须 >= 1，得到 {r}")

    m = vol.shape[axis]
    n = target_n if target_n is not None else int(round(m * r))
    if n < m:
        raise ValueError(f"target_n={n} 小于当前长度 {m}")

    F = np.fft.fftshift(np.fft.fft(vol, axis=axis), axes=axis)

    # 把 m 个系数居中放入长度 n 的数组，两端（高频）补零。
    # 关键：DC 必须精确落在 n//2，否则奇偶长度组合会产生半个采样点错位。
    out_shape = list(vol.shape)
    out_shape[axis] = n
    Fpad = np.zeros(out_shape, dtype=np.complex128)
    left = n // 2 - m // 2  # 源 DC 在 m//2，目标 DC 在 n//2
    sl = [slice(None)] * 3
    sl[axis] = slice(left, left + m)
    Fpad[tuple(sl)] = F

    # 幅度缩放：频域零填充上采样需乘 n/m（=r），恢复原信号幅值
    Fpad = Fpad * (n / m)

    Fpad = np.fft.ifftshift(Fpad, axes=axis)
    out = np.fft.ifft(Fpad, axis=axis).real
    return out


def lowpass_along_axis(
    array: np.ndarray,
    axis: int,
    r: float,
    filter_name: str = "gaussian",
) -> np.ndarray:
    """沿 axis 做一维低通（不降采样），返回低通后的实值数组。

    用于 SMORE 自监督训练：在 in-plane 切片上沿某一方向施加与 through-plane
    退化相同形式的一维模糊核，构造训练输入（见 Day 3）。

    Parameters
    ----------
    array : np.ndarray
        任意维度数组（通常 2D 切片或 3D 体数据），不会被修改。
    axis : int
        模糊方向。
    r : float
        与 through-plane 退化一致的倍率（f_cut = Nyquist/r）。
    filter_name : {"gaussian", "box", "fermi"}
    """
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim < 1 or not (0 <= axis < arr.ndim):
        raise ValueError(f"非法 axis={axis} for ndim={arr.ndim}")
    n = arr.shape[axis]
    mask = _lowpass_mask(n, r, filter_name)
    F = np.fft.fft(arr, axis=axis)
    F = np.fft.fftshift(F, axes=axis)
    F = F * _broadcast_mask(mask, axis, arr.shape)
    F = np.fft.ifftshift(F, axes=axis)
    return np.fft.ifft(F, axis=axis).real


def kspace_power_spectrum(volume: np.ndarray, axis: int) -> np.ndarray:
    """沿 axis 的一维平均功率谱 |FFT|²（用于频谱示意图）。

    返回 fftshift 顺序的均值谱，长度等于该轴体素数。
    """
    vol = np.asarray(volume, dtype=np.float64)
    F = np.fft.fftshift(np.fft.fft(vol, axis=axis), axes=axis)
    power = np.abs(F) ** 2
    # 对另外两个轴求平均
    axes_to_mean = tuple(a for a in range(3) if a != axis)
    return power.mean(axis=axes_to_mean)
