"""MRI 数据读取、元数据检查与强度归一化工具。

本模块属于 MRI 超分辨率项目的 Day 1 数据管线，只负责 I/O 与预处理，
不包含任何网络模型或超分辨率重建逻辑。

设计原则
--------
1. 原始文件只读：``load_mri`` 不会修改磁盘上的 NIfTI 文件。
2. 空间元数据（体素间距、方向、Affine）全部来自 NiBabel 读取的
   NIfTI header / affine，而不是从数组形状猜测。
3. 读取与归一化分离：``normalize_mri`` 始终返回新数组，不修改输入。
4. 三视图函数假定输入为 canonical RAS+ 方向的数据
   （通过 ``load_mri(..., canonical=True)`` 获得），这样轴的含义是固定的：
   axis 0 = Right, axis 1 = Anterior, axis 2 = Superior。

注意：体素间距（voxel spacing）是重建网格的采样间隔，不完全等价于
真实采集分辨率（层厚、层间距、部分容积效应等都会影响后者）。
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np

SUPPORTED_SUFFIXES = (".nii", ".nii.gz")


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def _validate_path(path: str | Path) -> Path:
    """检查文件存在且是受支持的 NIfTI 文件。"""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"MRI 文件不存在: {p}")
    if not p.is_file():
        raise ValueError(f"路径不是文件: {p}")
    # 注意 .nii.gz 需要 endswith 检查，不能用 Path.suffix（只会得到 .gz）
    if not p.name.lower().endswith(SUPPORTED_SUFFIXES):
        raise ValueError(
            f"不支持的文件格式: {p.name}；仅支持 {SUPPORTED_SUFFIXES}。"
        )
    return p


def _axcodes(affine: np.ndarray) -> str:
    """把 affine 转成三字母方向码，例如 'RAS'。无法判断时用 '?'。"""
    codes = nib.aff2axcodes(np.asarray(affine, dtype=np.float64))
    return "".join(code if code is not None else "?" for code in codes)


def _header_info(img: nib.spatialimages.SpatialImage) -> Dict[str, Any]:
    """提取常用的 NIfTI header 信息，全部转为可 JSON 序列化的类型。"""
    hdr = img.header
    try:
        xyzt_units = hdr.get_xyzt_units()
    except Exception:  # pragma: no cover - 非标准 header 的兜底
        xyzt_units = ("unknown", "unknown")

    descrip = hdr.get("descrip", b"")
    if isinstance(descrip, bytes):
        descrip = descrip.decode("latin-1", errors="replace")

    return {
        "qform_code": int(hdr["qform_code"]),
        "sform_code": int(hdr["sform_code"]),
        "xyzt_units": [str(xyzt_units[0]), str(xyzt_units[1])],
        "pixdim": [float(x) for x in np.asarray(hdr["pixdim"]).ravel()],
        "descrip": str(descrip).strip("\x00").strip(),
        "data_dtype_header": str(hdr.get_data_dtype()),
    }


def _spacing_from_header(img: nib.spatialimages.SpatialImage) -> Tuple[float, float, float]:
    """从 header zooms 读取前三个体素间距（mm）。"""
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=np.float64)
    if zooms.shape != (3,):
        raise ValueError(f"无法从 header 读取三维 voxel spacing，得到: {zooms}")
    return tuple(float(z) for z in zooms)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class MRIVolume:
    """一次 MRI 读取的结果：三维体数据 + 空间元数据。"""

    path: Path
    data: np.ndarray
    affine: np.ndarray
    voxel_spacing: Tuple[float, float, float]
    orientation: str
    shape: Tuple[int, ...]
    dtype: str
    canonical: bool = False
    original_shape: Optional[Tuple[int, ...]] = None
    original_orientation: Optional[str] = None
    header_info: Dict[str, Any] = field(default_factory=dict)

    def metadata(self) -> Dict[str, Any]:
        """返回可 JSON 序列化的元数据（不包含体数据本身）。"""
        meta = asdict(self)
        meta.pop("data")
        meta["path"] = str(self.path)
        meta["affine"] = np.asarray(self.affine, dtype=float).tolist()
        meta["shape"] = list(self.shape)
        if self.original_shape is not None:
            meta["original_shape"] = list(self.original_shape)
        return meta


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------
def load_mri(
    path: str | Path,
    *,
    canonical: bool = False,
    check_finite: bool = True,
    dtype: Optional[np.dtype] = None,
) -> MRIVolume:
    """读取一个 NIfTI 文件，返回 :class:`MRIVolume`。

    Parameters
    ----------
    path : str | Path
        ``.nii`` 或 ``.nii.gz`` 文件。
    canonical : bool
        是否用 ``nib.as_closest_canonical`` 重排为 RAS+ 方向。
        该操作只改变内存中的数据视图和 affine，不写回磁盘。
    check_finite : bool
        是否检查 NaN/Inf；大体积数据可关闭以节省时间。
    dtype : np.dtype | None
        可选的类型转换（例如转 float32）。默认保留原始类型。

    Raises
    ------
    FileNotFoundError
        文件不存在。
    ValueError
        格式不支持、维度不是三维、或包含非有限数值。
    """
    p = _validate_path(path)
    img = nib.load(str(p))

    original_shape = tuple(int(s) for s in img.shape)
    original_orientation = _axcodes(img.affine)

    if canonical:
        # 只影响内存中的视图，原始文件保持不变
        img = nib.as_closest_canonical(img)

    data = np.asanyarray(img.dataobj)

    # 容忍 (X, Y, Z, 1) 形式的四维文件（例如某些工具导出的单通道数据）
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]

    if data.ndim != 3:
        raise ValueError(
            f"期望三维 MRI 体数据，但 {p.name} 的 shape={data.shape} "
            f"(ndim={data.ndim})。本阶段只处理三维 T1 体数据。"
        )
    if min(data.shape) < 2:
        raise ValueError(f"非法三维维度: {p.name} shape={data.shape}")

    if dtype is not None and data.dtype != np.dtype(dtype):
        data = data.astype(dtype)

    if check_finite and not np.all(np.isfinite(data)):
        n_bad = int(data.size - np.count_nonzero(np.isfinite(data)))
        raise ValueError(
            f"{p.name} 包含 {n_bad} 个 NaN/Inf 体素，拒绝作为有效 MRI 数据读取。"
        )

    affine = np.asarray(img.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError(f"{p.name} 的 affine 非法: {affine}")
    if abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError(f"{p.name} 的 affine 前 3x3 矩阵不可逆（退化方向）。")

    spacing = _spacing_from_header(img)

    return MRIVolume(
        path=p,
        data=data,
        affine=affine,
        voxel_spacing=spacing,
        orientation=_axcodes(affine),
        shape=tuple(int(s) for s in data.shape),
        dtype=str(data.dtype),
        canonical=bool(canonical),
        original_shape=original_shape if canonical else None,
        original_orientation=original_orientation if canonical else None,
        header_info=_header_info(img),
    )


# ---------------------------------------------------------------------------
# 检查
# ---------------------------------------------------------------------------
def _percentiles(values: np.ndarray, qs: Sequence[float]) -> list[float]:
    if values.size == 0:
        return [float("nan") for _ in qs]
    return [float(v) for v in np.percentile(values, qs)]


def inspect_mri(
    path: str | Path,
    *,
    percentiles: Sequence[float] = (1.0, 50.0, 99.0),
    background_threshold: float = 0.0,
    compute_canonical: bool = True,
) -> Dict[str, Any]:
    """读取并检查一个 MRI 文件，返回结构化字典（可直接写入 JSON）。

    检查内容包括：shape、voxel spacing、方向、affine、强度统计、
    NaN/Inf、是否为有效三维医学图像，以及 canonical RAS+ 重排前后的对比。
    """
    vol = load_mri(path, canonical=False, check_finite=False)
    data = vol.data

    all_finite = np.isfinite(data)
    n_nonfinite = int(data.size - np.count_nonzero(all_finite))
    finite_values = data[all_finite]  # 仅用于统计，原始数组不被修改

    is_float = np.issubdtype(data.dtype, np.floating)
    has_nan = bool(np.isnan(data).any()) if is_float else False
    has_inf = bool(np.isinf(data).any()) if is_float else False

    fg_mask = data > background_threshold
    fg_values = data[fg_mask]
    n_fg = int(fg_mask.sum())

    det = float(np.linalg.det(np.asarray(vol.affine)[:3, :3]))
    is_valid_3d = bool(
        data.ndim == 3
        and min(data.shape) >= 2
        and np.all(np.isfinite(vol.affine))
        and abs(det) > 1e-12
        and n_nonfinite == 0
    )

    report: Dict[str, Any] = {
        "path": str(vol.path),
        "file_name": vol.path.name,
        "file_size_bytes": int(vol.path.stat().st_size),
        "shape": list(vol.shape),
        "voxel_spacing_mm": [round(v, 6) for v in vol.voxel_spacing],
        "orientation": vol.orientation,
        "affine": np.round(vol.affine, 6).tolist(),
        "dtype": vol.dtype,
        "header": vol.header_info,
        "intensity": {
            "min": float(np.min(finite_values)) if finite_values.size else float("nan"),
            "max": float(np.max(finite_values)) if finite_values.size else float("nan"),
            "percentile_levels": [float(q) for q in percentiles],
            "percentiles_all": _percentiles(finite_values, percentiles),
            "percentiles_foreground": _percentiles(fg_values, percentiles),
            "background_threshold": float(background_threshold),
            "foreground_voxels": n_fg,
            "total_voxels": int(data.size),
            "foreground_fraction": float(n_fg / data.size) if data.size else 0.0,
        },
        "has_nan": has_nan,
        "has_inf": has_inf,
        "nonfinite_voxels": n_nonfinite,
        "is_valid_3d": is_valid_3d,
        "affine_determinant": det,
    }

    if compute_canonical:
        # 重新用官方 API 获取 RAS+ 视图，只读取已存在的元数据
        canon_img = nib.as_closest_canonical(nib.load(str(vol.path)))
        canon_shape = tuple(int(s) for s in canon_img.shape)
        canon_affine = np.asarray(canon_img.affine, dtype=np.float64)
        canon_zooms = tuple(float(z) for z in canon_img.header.get_zooms()[:3])
        report["canonical_ras"] = {
            "shape": list(canon_shape),
            "voxel_spacing_mm": [round(z, 6) for z in canon_zooms],
            "orientation": _axcodes(canon_affine),
            "affine": np.round(canon_affine, 6).tolist(),
            "shape_changed": canon_shape != tuple(vol.shape),
            "orientation_changed": _axcodes(canon_affine) != vol.orientation,
            "note": (
                "canonical 视图只在内存中生成，原始文件未被修改。"
                "若 shape 改变说明发生了轴置换（axis permutation）。"
            ),
        }

    return report


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------
def normalize_mri(
    volume: np.ndarray,
    *,
    method: str = "percentile",
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.5,
    background_threshold: float = 0.0,
    clip: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """对三维 MRI 做可复用的强度归一化，返回新的 float32 数组。

    为什么用分位数而不是简单 min-max？
    ------------------------------------
    T1 MRI 背景为 0，亮度分布高度偏斜，少数极亮体素（脂肪、血管、伪影）
    会压缩脑组织的动态范围。因此默认按**前景**（> background_threshold）
    的第 1 与第 99.5 百分位做线性拉伸，并把背景体素裁剪到 0：
    背景仍然保持为 0，脑组织获得稳定对比度。

    该方法对每名受试者独立进行，不需要数据集级统计，适用于自监督训练。

    Parameters
    ----------
    volume : np.ndarray
        三维输入数组（不会被修改）。
    method : {"percentile", "minmax", "zscore"}
        - percentile: 前景分位数线性拉伸到 [0, 1]（默认）
        - minmax: 全体体素 min-max 拉伸到 [0, 1]
        - zscore: 前景均值/标准差标准化（输出可为负）
    lower_percentile, upper_percentile : float
        分位数归一化的下/上分位数（0-100）。
    background_threshold : float
        小于等于该值视为背景；T1 MRI 通常为 0。
    clip : bool
        是否把 percentile 结果裁剪到 [0, 1]。
    eps : float
        防止除零的最小动态范围。

    Returns
    -------
    np.ndarray
        float32 新数组，shape 与输入相同。

    Raises
    ------
    ValueError
        输入不是三维、或包含 NaN/Inf、或参数非法。
    """
    arr = np.asarray(volume)
    if arr.ndim != 3:
        raise ValueError(f"normalize_mri 期望三维数组，得到 shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("输入数组包含 NaN/Inf，无法归一化。请先检查数据。")
    if method not in {"percentile", "minmax", "zscore"}:
        raise ValueError(f"未知归一化方法: {method}")
    if not (0.0 <= lower_percentile < upper_percentile <= 100.0):
        raise ValueError(
            f"非法分位数范围: [{lower_percentile}, {upper_percentile}]"
        )

    # copy=True 保证不修改调用者的数组
    out = np.array(arr, dtype=np.float32, copy=True)
    fg_mask = out > background_threshold

    if method == "zscore":
        values = out[fg_mask] if fg_mask.any() else out.ravel()
        mean = float(values.mean())
        std = float(values.std())
        if std < eps:
            warnings.warn("输入近似常数，zscore 归一化返回全零数组。", RuntimeWarning)
            return np.zeros_like(out, dtype=np.float32)
        # z-score 输出不以 [0,1] 为范围，clip 参数仅对 percentile/minmax 生效
        out = (out - mean) / std
        return out.astype(np.float32, copy=False)

    if method == "minmax":
        lo, hi = float(out.min()), float(out.max())
    else:  # percentile
        values = out[fg_mask] if fg_mask.any() else out.ravel()
        lo, hi = (float(v) for v in np.percentile(values, [lower_percentile, upper_percentile]))

    if hi - lo < eps:
        warnings.warn(
            f"输入动态范围过小 (hi={hi}, lo={lo})，归一化返回全零数组。",
            RuntimeWarning,
        )
        return np.zeros_like(out, dtype=np.float32)

    out = (out - lo) / (hi - lo)
    if clip:
        np.clip(out, 0.0, 1.0, out=out)
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# 三视图切片（仅用于可视化 / 后续配准检查）
# ---------------------------------------------------------------------------
def orthogonal_slices(
    volume: np.ndarray,
    voxel_spacing: Sequence[float],
    indices: Optional[Sequence[int]] = None,
) -> Dict[str, Dict[str, Any]]:
    """提取 axial / coronal / sagittal 三个正交切片。

    输入必须是 canonical RAS+ 数据：axis 0=R, axis 1=A, axis 2=S。
    返回的二维切片已经转置为 ``imshow`` 约定：行 = 图像纵向，列 = 图像横向。

    Returns
    -------
    dict
        每个方向包含 ``slice``（2D 数组）、``extent``（mm）、
        ``x_axis``/``y_axis``（解剖方向）、``index``（切片序号）。
    """
    arr = np.asarray(volume)
    if arr.ndim != 3:
        raise ValueError(f"orthogonal_slices 期望三维数组，得到 {arr.shape}")
    spacing = tuple(float(s) for s in voxel_spacing)
    if len(spacing) != 3 or any(s <= 0 for s in spacing):
        raise ValueError(f"非法 voxel spacing: {spacing}")

    nx, ny, nz = arr.shape
    if indices is None:
        ix, iy, iz = nx // 2, ny // 2, nz // 2
    else:
        if len(indices) != 3:
            raise ValueError("indices 必须是三个整数 (i_R, i_A, i_S)")
        ix, iy, iz = (int(i) for i in indices)
        if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
            raise IndexError(f"indices={indices} 超出 shape={arr.shape}")

    dx, dy, dz = spacing
    return {
        "axial": {
            "slice": arr[:, :, iz].T,  # rows=A, cols=R
            "extent": (0.0, nx * dx, 0.0, ny * dy),
            "x_axis": "R -> right",
            "y_axis": "A -> anterior",
            "index": int(iz),
            "index_axis": "S",
        },
        "coronal": {
            "slice": arr[:, iy, :].T,  # rows=S, cols=R
            "extent": (0.0, nx * dx, 0.0, nz * dz),
            "x_axis": "R -> right",
            "y_axis": "S -> superior",
            "index": int(iy),
            "index_axis": "A",
        },
        "sagittal": {
            "slice": arr[ix, :, :].T,  # rows=S, cols=A
            "extent": (0.0, ny * dy, 0.0, nz * dz),
            "x_axis": "A -> anterior",
            "y_axis": "S -> superior",
            "index": int(ix),
            "index_axis": "R",
        },
    }
