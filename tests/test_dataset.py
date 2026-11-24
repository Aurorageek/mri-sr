"""src/dataset.py 的单元测试。

重要说明
--------
本文件使用**人工合成数组**测试函数行为（形状、归一化、报错、方向重排），
目的是验证代码逻辑，不代表真实 IXI MRI 数据验证。
真实数据验证由 ``scripts/inspect_mri.py`` 在真实 .nii.gz 上完成。

运行::

    python tests/test_dataset.py
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (
    inspect_mri,
    load_mri,
    normalize_mri,
    orthogonal_slices,
)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    """装饰器：执行测试函数并记录 PASS/FAIL。"""

    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                fn(*args, **kwargs)
                RESULTS.append((name, True, ""))
                print(f"PASS  {name}")
            except Exception as exc:  # noqa: BLE001 - 测试框架需要捕获所有异常
                RESULTS.append((name, False, repr(exc)))
                print(f"FAIL  {name}\n      {exc!r}")

        return wrapper

    return decorator


def make_synthetic_volume(shape=(20, 30, 40), seed=0) -> np.ndarray:
    """生成带背景的合成 int16 体数据（仅用于单元测试，不是真实 MRI）。"""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 1000, size=shape, dtype=np.int16)
    data[:2, :, :] = 0  # 人为制造背景
    data[2:5] = 0
    return data


def save_nifti(data: np.ndarray, affine: np.ndarray, path: Path) -> Path:
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(path))
    return path


@check("load_mri: 读取合成 NIfTI 并返回正确的 shape/spacing/orientation")
def test_load_mri(tmp: Path):
    data = make_synthetic_volume()
    affine = np.diag([2.0, 1.5, 3.0, 1.0])  # 正定对角 -> RAS+
    path = save_nifti(data, affine, tmp / "synthetic_ras.nii.gz")

    vol = load_mri(path)
    assert vol.data.shape == data.shape, vol.data.shape
    np.testing.assert_array_equal(vol.data, data)
    assert vol.orientation == "RAS", vol.orientation
    np.testing.assert_allclose(vol.voxel_spacing, (2.0, 1.5, 3.0), atol=1e-5)
    assert vol.canonical is False
    assert vol.dtype == "int16", vol.dtype
    np.testing.assert_allclose(vol.affine, affine)


@check("load_mri: canonical=True 对 LAS 数据做轴翻转且不修改原文件")
def test_load_mri_canonical(tmp: Path):
    data = make_synthetic_volume(shape=(16, 20, 24), seed=1)
    affine = np.diag([-2.0, 1.5, 3.0, 1.0])  # x 轴为 L -> LAS
    path = save_nifti(data, affine, tmp / "synthetic_las.nii")
    size_before = path.stat().st_size
    mtime_before = path.stat().st_mtime_ns

    raw = load_mri(path)
    assert raw.orientation == "LAS", raw.orientation

    canonical = load_mri(path, canonical=True)
    assert canonical.canonical is True
    assert canonical.orientation == "RAS", canonical.orientation
    assert canonical.original_orientation == "LAS"
    assert canonical.original_shape == data.shape
    np.testing.assert_array_equal(canonical.data, data[::-1, :, :])
    # 原始文件必须保持不变
    assert path.stat().st_size == size_before
    assert path.stat().st_mtime_ns == mtime_before


@check("inspect_mri: 返回完整元数据与强度统计")
def test_inspect_mri(tmp: Path):
    data = make_synthetic_volume()
    path = save_nifti(data, np.diag([2.0, 1.5, 3.0, 1.0]), tmp / "synthetic_inspect.nii.gz")
    rep = inspect_mri(path)

    assert rep["shape"] == list(data.shape)
    np.testing.assert_allclose(rep["voxel_spacing_mm"], [2.0, 1.5, 3.0], atol=1e-5)
    assert rep["orientation"] == "RAS"
    assert rep["is_valid_3d"] is True
    assert rep["has_nan"] is False and rep["has_inf"] is False
    assert rep["nonfinite_voxels"] == 0
    assert rep["intensity"]["min"] == float(data.min())
    assert rep["intensity"]["max"] == float(data.max())
    p1, p50, p99 = rep["intensity"]["percentiles_all"]
    assert p1 <= p50 <= p99
    assert rep["intensity"]["foreground_voxels"] == int((data > 0).sum())
    canon = rep["canonical_ras"]
    assert canon["orientation"] == "RAS"
    assert canon["shape"] == list(data.shape)
    assert canon["shape_changed"] is False


@check("inspect_mri: LAS 数据记录 canonical 重排前后的方向")
def test_inspect_canonical_change(tmp: Path):
    data = make_synthetic_volume(shape=(10, 12, 14), seed=2)
    path = save_nifti(data, np.diag([-2.0, 1.5, 3.0, 1.0]), tmp / "synthetic_las_inspect.nii.gz")
    rep = inspect_mri(path)
    assert rep["orientation"] == "LAS"
    canon = rep["canonical_ras"]
    assert canon["orientation"] == "RAS"
    assert canon["orientation_changed"] is True
    assert canon["shape"] == list(data.shape)  # 只是翻转，不改变维数


@check("normalize_mri: 不修改输入、返回 float32、背景为 0、范围在 [0,1]")
def test_normalize_percentile(tmp: Path):
    data = make_synthetic_volume()
    before = data.copy()
    out = normalize_mri(data)

    np.testing.assert_array_equal(data, before)  # 输入未被修改
    assert out.dtype == np.float32
    assert not np.shares_memory(out, data)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.all(out[data == 0] == 0.0)  # 背景保持 0
    assert out.max() > 0.5  # 对比度确实被拉开


@check("normalize_mri: 常数输入不除零，返回全零并给出警告")
def test_normalize_constant():
    data = np.full((8, 8, 8), 7.0, dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = normalize_mri(data)
    assert np.all(out == 0.0)
    assert any(isinstance(w.message, RuntimeWarning) for w in caught)


@check("normalize_mri: 拒绝 NaN/Inf 输入")
def test_normalize_nonfinite():
    data = np.zeros((8, 8, 8), dtype=np.float32)
    data[3, 3, 3] = np.nan
    try:
        normalize_mri(data)
    except ValueError:
        return
    raise AssertionError("包含 NaN 的输入应当抛出 ValueError")


@check("normalize_mri: 拒绝非三维输入")
def test_normalize_wrong_ndim():
    try:
        normalize_mri(np.zeros((8, 8), dtype=np.float32))
    except ValueError:
        return
    raise AssertionError("二维输入应当抛出 ValueError")


@check("normalize_mri: minmax 与 zscore 方法可用")
def test_normalize_other_methods():
    data = make_synthetic_volume()
    mm = normalize_mri(data, method="minmax")
    assert abs(float(mm.min())) < 1e-6 and abs(float(mm.max()) - 1.0) < 1e-6

    z = normalize_mri(data, method="zscore")
    fg = z[data > 0]
    assert abs(float(fg.mean())) < 1e-4
    assert abs(float(fg.std()) - 1.0) < 1e-3


@check("错误处理: 文件缺失 / 格式不支持 / 非三维数据")
def test_error_handling(tmp: Path):
    try:
        load_mri(tmp / "does_not_exist.nii.gz")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("缺失文件应抛出 FileNotFoundError")

    bad = tmp / "not_mri.txt"
    bad.write_text("hello", encoding="utf-8")
    try:
        load_mri(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("不支持的格式应抛出 ValueError")

    data4d = np.zeros((8, 8, 8, 2), dtype=np.float32)
    path4d = save_nifti(data4d, np.eye(4), tmp / "synthetic_4d.nii.gz")
    try:
        load_mri(path4d)
    except ValueError as exc:
        assert "三维" in str(exc)
    else:
        raise AssertionError("四维数据应抛出 ValueError")

    data_nan = np.zeros((8, 8, 8), dtype=np.float32)
    data_nan[0, 0, 0] = np.inf
    path_nan = save_nifti(data_nan, np.eye(4), tmp / "synthetic_inf.nii.gz")
    try:
        load_mri(path_nan)
    except ValueError as exc:
        assert "NaN/Inf" in str(exc)
    else:
        raise AssertionError("含 Inf 的数据应抛出 ValueError")


@check("orthogonal_slices: 三视图形状、extent 与索引正确")
def test_orthogonal_slices(tmp: Path):
    data = make_synthetic_volume(shape=(16, 20, 24), seed=3)
    affine = np.diag([2.0, 1.5, 3.0, 1.0])
    path = save_nifti(data, affine, tmp / "synthetic_slices.nii.gz")
    vol = load_mri(path, canonical=True)

    nx, ny, nz = vol.shape
    dx, dy, dz = vol.voxel_spacing
    views = orthogonal_slices(vol.data, vol.voxel_spacing)

    np.testing.assert_array_equal(views["axial"]["slice"], vol.data[:, :, nz // 2].T)
    np.testing.assert_array_equal(views["coronal"]["slice"], vol.data[:, ny // 2, :].T)
    np.testing.assert_array_equal(views["sagittal"]["slice"], vol.data[nx // 2, :, :].T)

    assert views["axial"]["slice"].shape == (ny, nx)
    assert views["coronal"]["slice"].shape == (nz, nx)
    assert views["sagittal"]["slice"].shape == (nz, ny)

    np.testing.assert_allclose(views["axial"]["extent"], (0, nx * dx, 0, ny * dy))
    np.testing.assert_allclose(views["coronal"]["extent"], (0, nx * dx, 0, nz * dz))
    np.testing.assert_allclose(views["sagittal"]["extent"], (0, ny * dy, 0, nz * dz))

    chosen = orthogonal_slices(vol.data, vol.voxel_spacing, indices=(1, 2, 3))
    assert chosen["axial"]["index"] == 3
    assert chosen["coronal"]["index"] == 2
    assert chosen["sagittal"]["index"] == 1

    try:
        orthogonal_slices(vol.data, vol.voxel_spacing, indices=(999, 0, 0))
    except IndexError:
        pass
    else:
        raise AssertionError("越界索引应抛出 IndexError")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mri_sr_unit_") as td:
        tmp = Path(td)
        test_load_mri(tmp)
        test_load_mri_canonical(tmp)
        test_inspect_mri(tmp)
        test_inspect_canonical_change(tmp)
        test_normalize_percentile(tmp)
        test_normalize_constant()
        test_normalize_nonfinite()
        test_normalize_wrong_ndim()
        test_normalize_other_methods()
        test_error_handling(tmp)
        test_orthogonal_slices(tmp)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = len(RESULTS) - passed
    print("\n" + "-" * 60)
    print(f"单元测试汇总（合成数组）: {passed} passed, {failed} failed, {len(RESULTS)} total")
    if failed:
        for name, ok, err in RESULTS:
            if not ok:
                print(f"  FAILED: {name} -> {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
