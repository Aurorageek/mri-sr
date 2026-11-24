"""图像质量评价：PSNR、SSIM 与脑组织评价 mask（Day 2）。

设计原则
--------

* **统一强度域**：PSNR/SSIM 直接在 HR 与重建结果共有的原始强度上计算，
  数据范围（data_range）由 HR 前景动态范围决定，对所有受试者、所有方法
  使用同一规则，**不逐方法单独调窗**，避免人为抬高指标。
* **只在脑组织有效区域计算**：用 HR 生成评价 mask，排除背景空气、颅骨边缘
  与部分容积体素，使指标反映脑实质的重建质量。
* **空间对齐**：要求 ref 与 test 形状完全一致（重建结果已在 HR 网格上）。

评价 mask 生成方法（记录在案，见 README 与 metrics JSON）：

1. ``thr = 0.01 * percentile99.9(HR)``（稳健前景阈值，对极亮体素不敏感）；
2. ``mask0 = HR > thr``；
3. 对 mask0 做两次 3D 二值腐蚀（结构元素 3×3×3），去除颅骨/边缘部分容积体素。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity as _ssim


def brain_mask(
    reference: np.ndarray,
    *,
    threshold_frac: float = 0.01,
    erosion_iterations: int = 2,
) -> np.ndarray:
    """由参考 HR 图像生成脑组织评价 mask（bool）。

    Parameters
    ----------
    reference : np.ndarray
        三维参考图像（HR）。
    threshold_frac : float
        前景阈值 = threshold_frac * P99.9(reference)。
    erosion_iterations : int
        3D 二值腐蚀次数，去除边缘部分容积体素。
    """
    ref = np.asarray(reference, dtype=np.float64)
    if ref.ndim != 3:
        raise ValueError(f"brain_mask 期望三维数组，得到 {ref.shape}")
    if not np.all(np.isfinite(ref)):
        raise ValueError("参考图像包含 NaN/Inf")

    p999 = float(np.percentile(ref, 99.9))
    thr = threshold_frac * p999
    mask0 = ref > thr
    if not mask0.any():
        raise ValueError("评价 mask 为空：参考图像前景体素不足。")
    if erosion_iterations > 0:
        mask0 = ndimage.binary_erosion(mask0, iterations=erosion_iterations)
    return mask0


def _check_alignment(ref: np.ndarray, test: np.ndarray, mask: np.ndarray) -> None:
    if ref.shape != test.shape:
        raise ValueError(f"ref 与 test 形状不一致: {ref.shape} vs {test.shape}")
    if mask.shape != ref.shape:
        raise ValueError(f"mask 形状不一致: {mask.shape} vs {ref.shape}")
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(test)):
        raise ValueError("输入包含 NaN/Inf")


def mse(ref: np.ndarray, test: np.ndarray, mask: np.ndarray) -> float:
    """脑组织区域的均方误差。"""
    _check_alignment(ref, test, mask)
    r = np.asarray(ref, dtype=np.float64)
    t = np.asarray(test, dtype=np.float64)
    return float(np.mean((r[mask] - t[mask]) ** 2))


def mae(ref: np.ndarray, test: np.ndarray, mask: np.ndarray) -> float:
    """脑组织区域的平均绝对误差（MAE = mean |ref - test|）。"""
    _check_alignment(ref, test, mask)
    r = np.asarray(ref, dtype=np.float64)
    t = np.asarray(test, dtype=np.float64)
    return float(np.mean(np.abs(r[mask] - t[mask])))


def psnr(ref: np.ndarray, test: np.ndarray, mask: np.ndarray) -> float:
    """脑组织区域的峰值信噪比（dB）。

    data_range 取参考图像在 mask 内的动态范围（max - min）。
    """
    _check_alignment(ref, test, mask)
    r = np.asarray(ref, dtype=np.float64)
    t = np.asarray(test, dtype=np.float64)
    err = mse(r, t, mask)
    if err == 0.0:
        return float("inf")
    data_range = float(r[mask].max() - r[mask].min())
    if data_range == 0.0:
        return float("nan")
    return float(10.0 * np.log10(data_range**2 / err))


def ssim(
    ref: np.ndarray,
    test: np.ndarray,
    mask: np.ndarray,
    *,
    win_size: int = 7,
    data_range: float | None = None,
) -> float:
    """脑组织区域的 3D SSIM。

    在 mask 的包围盒内计算（裁剪掉纯背景区域以加速），并对裁剪后的 mask 做
    1 体素腐蚀以削弱窗口跨 mask 边界的效应。
    """
    _check_alignment(ref, test, mask)
    r = np.asarray(ref, dtype=np.float64)
    t = np.asarray(test, dtype=np.float64)

    # 包围盒裁剪
    idx = np.argwhere(mask)
    (z0, y0, x0), (z1, y1, x1) = idx.min(axis=0), idx.max(axis=0)
    margin = max(win_size, 4)
    z0 = max(0, z0 - margin); y0 = max(0, y0 - margin); x0 = max(0, x0 - margin)
    z1 = min(r.shape[0], z1 + margin + 1)
    y1 = min(r.shape[1], y1 + margin + 1)
    x1 = min(r.shape[2], x1 + margin + 1)

    rc = r[z0:z1, y0:y1, x0:x1]
    tc = t[z0:z1, y0:y1, x0:x1]
    mc = mask[z0:z1, y0:y1, x0:x1]
    if mc.sum() > 0 and win_size > 1:
        mc = ndimage.binary_erosion(mc, iterations=1)

    if data_range is None:
        data_range = float(rc[mc].max() - rc[mc].min()) if mc.any() else float(r.max() - r.min())
    if data_range <= 0:
        return float("nan")

    # full=True 返回逐窗口 SSIM 图（与输入同形），再按 mask 加权平均
    ssim_map = _ssim(
        rc, tc, win_size=win_size, data_range=data_range,
        gaussian_weights=False, use_sample_covariance=False, full=True,
    )[1]
    if mc.any():
        return float(np.mean(ssim_map[mc]))
    return float(np.mean(ssim_map))


def evaluate(
    ref: np.ndarray,
    test: np.ndarray,
    mask: np.ndarray,
    *,
    win_size: int = 7,
) -> dict:
    """一次算出 MSE / PSNR / SSIM，返回字典。"""
    return {
        "mse": mse(ref, test, mask),
        "psnr_db": psnr(ref, test, mask),
        "ssim": ssim(ref, test, mask, win_size=win_size),
        "mask_voxels": int(np.count_nonzero(mask)),
    }


def sobel_axis_mae(
    ref: np.ndarray,
    test: np.ndarray,
    mask: np.ndarray,
    axis: int,
    spacing: float = 1.0,
) -> float:
    """沿单个空间轴的单轴 Sobel 边缘误差（可除以该轴 voxel spacing）。

    ``scipy.ndimage.sobel`` 指定 axis 时只计算该轴的 Sobel 响应。
    """
    _check_alignment(ref, test, mask)
    r = np.asarray(ref, dtype=np.float64)
    t = np.asarray(test, dtype=np.float64)
    g_r = ndimage.sobel(r, axis=axis) / spacing
    g_t = ndimage.sobel(t, axis=axis) / spacing
    return float(np.mean(np.abs(g_r[mask] - g_t[mask])))


def through_plane_gradient_mae(
    ref: np.ndarray,
    test: np.ndarray,
    mask: np.ndarray,
    axis: int = 0,
    spacing: float = 1.0,
) -> float:
    """through-plane 方向的结构误差：沿退化轴（默认 R=axis0）的 Sobel 边缘误差。

    Day 5 确认退化轴为 R（canonical axis 0，voxel spacing 1.2mm）。
    """
    return sobel_axis_mae(ref, test, mask, axis=axis, spacing=spacing)


def gradient_3d_mae(
    ref: np.ndarray,
    test: np.ndarray,
    mask: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """三维空间梯度幅值误差，考虑各轴实际 voxel spacing。

    定义：::

        g_axis_i = sobel(vol, axis=i) / spacing_i
        mag = sqrt(sum_i g_axis_i^2)
        error = mean_{mask}( | mag(ref) - mag(test) | )
    """
    _check_alignment(ref, test, mask)
    r = np.asarray(ref, dtype=np.float64)
    t = np.asarray(test, dtype=np.float64)
    mag_r = np.sqrt(sum((ndimage.sobel(r, axis=i) / spacing[i]) ** 2 for i in range(3)))
    mag_t = np.sqrt(sum((ndimage.sobel(t, axis=i) / spacing[i]) ** 2 for i in range(3)))
    return float(np.mean(np.abs(mag_r[mask] - mag_t[mask])))


def gradient_mae(
    ref: np.ndarray,
    test: np.ndarray,
    mask: np.ndarray,
) -> float:
    """【历史遗留指标】仅沿 S 轴（axis=-1）的单轴 Sobel 边缘误差。

    注意：Day 4/5 报告的 "Gradient MAE" 即此定义——它既不是三维梯度幅值，
    也没有对准 through-plane（R）方向。请改用 :func:`through_plane_gradient_mae`
    或 :func:`gradient_3d_mae`。
    """
    return sobel_axis_mae(ref, test, mask, axis=-1)


def evaluate_with_edge(
    ref: np.ndarray,
    test: np.ndarray,
    mask: np.ndarray,
    *,
    win_size: int = 7,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    through_plane_axis: int = 0,
) -> dict:
    """PSNR / SSIM / MAE / 边缘误差 一次算齐。

    返回：
    - ``mae``：真实平均绝对误差（Day 6 修正）。
    - ``gradient_mae``：历史遗留的 S 轴 Sobel 误差（保留供对照，明确标注）。
    - ``through_plane_gradient_mae``：退化方向（默认 R=axis0）边缘误差。
    - ``gradient_3d_mae``：考虑 voxel spacing 的三维梯度幅值误差。

    spacing 为 canonical RAS+ 的三轴 voxel spacing（mm），默认各向同性 (1,1,1)。
    """
    base = evaluate(ref, test, mask, win_size=win_size)
    base["mae"] = mae(ref, test, mask)
    base["gradient_mae"] = gradient_mae(ref, test, mask)
    base["through_plane_gradient_mae"] = through_plane_gradient_mae(
        ref, test, mask, axis=through_plane_axis, spacing=spacing[through_plane_axis]
    )
    base["gradient_3d_mae"] = gradient_3d_mae(ref, test, mask, spacing=spacing)
    return base
