"""src/degradation.py 与 src/metrics.py 的单元测试（合成数组）。

验证的是算子的数学正确性，不代表真实 MRI 数据；真实数据端到端由
scripts/eval_interpolation.py 完成。运行::

    python tests/test_degradation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.degradation import (
    degrade_through_plane,
    kspace_power_spectrum,
    sinc_interpolate,
)
from src.metrics import brain_mask, mse, psnr, ssim

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    def decorator(fn):
        def wrapper():
            try:
                fn()
                RESULTS.append((name, True, ""))
                print(f"PASS  {name}")
            except Exception as exc:  # noqa: BLE001
                RESULTS.append((name, False, repr(exc)))
                print(f"FAIL  {name}\n      {exc!r}")
        return wrapper
    return decorator


def bandlimited_1d(n=64, fmax=0.2, seed=0) -> np.ndarray:
    """生成严格带限（|f|<fmax）的一维信号，用于验证理想带限内插。"""
    rng = np.random.default_rng(seed)
    F = np.zeros(n, dtype=np.complex128)
    freqs = np.fft.fftfreq(n)
    inside = np.abs(freqs) < fmax
    F[inside] = rng.standard_normal(int(inside.sum())) + 1j * rng.standard_normal(int(inside.sum()))
    x = np.fft.ifft(F).real
    return x


def make_3d_from_1d(signal, axis, shape=(40, 48, 52)) -> np.ndarray:
    """把一维信号沿 axis 平铺成三维体（保证只有该轴有变化）。"""
    vol = np.zeros(shape, dtype=np.float64)
    sl = [slice(None)] * 3
    for i, v in enumerate(signal):
        sl[axis] = i
        vol[tuple(sl)] = v
    return vol


@check("r=1 时退化近似恒等（不下采样、不滤波）")
def test_r1_identity():
    vol = np.random.default_rng(0).standard_normal((16, 20, 24))
    deg = degrade_through_plane(vol, axis=0, r=1, filter_name="gaussian")
    np.testing.assert_allclose(deg.lr, vol, atol=1e-12)
    np.testing.assert_allclose(deg.filtered, vol, atol=1e-12)
    assert deg.lr.shape == vol.shape


@check("r=2 只移除退化轴高频，另外两轴不变")
def test_high_freq_removed_only_on_axis():
    rng = np.random.default_rng(1)
    vol = rng.standard_normal((32, 40, 44))
    axis = 0
    deg = degrade_through_plane(vol, axis=axis, r=2, filter_name="box")
    # 直接比较三维谱：退化轴高频被截断，低频带完全保留
    F = np.fft.fftshift(np.fft.fftn(vol))
    Ff = np.fft.fftshift(np.fft.fftn(deg.filtered))
    f = np.fft.fftshift(np.fft.fftfreq(vol.shape[axis]))
    high = np.abs(f) > 0.25 + 1e-3
    low = np.abs(f) <= 0.25 + 1e-3
    # 高频带能量应为 0（box 截断）
    assert np.abs(Ff[high]).max() < 1e-9, np.abs(Ff[high]).max()
    # 低频带（含另外两轴的全部频率）完全保留 -> 证明只动了退化轴
    np.testing.assert_allclose(Ff[low], F[low], atol=1e-9)
    # 下采样后长度减半
    assert deg.lr.shape[axis] == vol.shape[axis] // 2


@check("Sinc 插值对严格带限信号近似无损（band-limited 恢复）")
def test_sinc_bandlimited_exact():
    n = 128
    x = bandlimited_1d(n=n, fmax=0.2, seed=2)
    vol = make_3d_from_1d(x, axis=0, shape=(n, 8, 8))
    lr = degrade_through_plane(vol, axis=0, r=2, filter_name="box").lr  # box 保留带内
    rec = sinc_interpolate(lr, axis=0, r=2, target_n=n)
    # 带限信号在 HR 网格上应几乎完全恢复（理想 sinc）
    err = np.abs(rec - vol).max() / (np.abs(vol).max() + 1e-9)
    assert err < 1e-6, err


@check("Sinc 插值在偶数->奇数长度（150->75）下无半采样错位")
def test_sinc_odd_length_no_halfshift():
    # 回归测试：之前 left=pad//2 会在 n 偶数、m 奇数时把 DC 放错位置
    n = 150
    x = bandlimited_1d(n=n, fmax=0.2, seed=7)
    vol = make_3d_from_1d(x, axis=0, shape=(n, 8, 8))
    lr = degrade_through_plane(vol, axis=0, r=2, filter_name="box").lr  # 75 层
    assert lr.shape[0] == 75
    rec = sinc_interpolate(lr, axis=0, r=2, target_n=n)
    # 与 HR 对齐，逐点误差应极小
    err = np.abs(rec - vol).max() / (np.abs(vol).max() + 1e-9)
    assert err < 1e-6, err


@check("逆 FFT 虚部残差在数值精度量级")
def test_imag_residual_tiny():
    rng = np.random.default_rng(3)
    vol = rng.standard_normal((30, 30, 30))
    deg = degrade_through_plane(vol, axis=2, r=2, filter_name="gaussian")
    rec = sinc_interpolate(deg.lr, axis=2, r=2, target_n=30)
    imag_deg = np.abs(np.imag(np.fft.ifft(np.fft.fft(deg.lr, axis=2), axis=2))).max()
    imag_rec = np.abs(np.imag(np.fft.ifft(np.fft.fft(rec, axis=2), axis=2))).max()
    assert imag_deg < 1e-9, imag_deg
    assert imag_rec < 1e-9, imag_rec


@check("输出尺寸、间距、affine 更新正确")
def test_shape_spacing_affine():
    vol = np.random.default_rng(4).standard_normal((150, 20, 20))
    spacing = (1.2, 0.9375, 0.9375)
    affine = np.diag([1.2, 0.9375, 0.9375, 1.0])
    deg = degrade_through_plane(vol, axis=0, r=2, spacing=spacing, affine=affine, filter_name="gaussian")
    assert deg.lr.shape == (75, 20, 20)
    np.testing.assert_allclose(deg.lr_spacing, (2.4, 0.9375, 0.9375), atol=1e-6)
    np.testing.assert_allclose(deg.lr_affine[0, 0], 2.4, atol=1e-6)  # 轴0方向缩放2
    np.testing.assert_allclose(deg.lr_affine[1, 1], 0.9375, atol=1e-6)
    # sinc 插值恢复到 HR 长度
    rec = sinc_interpolate(deg.lr, axis=0, r=2, target_n=150)
    assert rec.shape == (150, 20, 20)


@check("metrics: 完全相同图像 PSNR=inf、SSIM=1")
def test_metrics_identity():
    rng = np.random.default_rng(5)
    img = rng.standard_normal((30, 30, 30)) + 5.0
    mask = brain_mask(img)
    assert psnr(img, img, mask) == float("inf")
    assert abs(ssim(img, img, mask) - 1.0) < 1e-4
    assert mse(img, img, mask) == 0.0


@check("metrics: 加噪图像 PSNR/SSIM 随噪声增大而下降")
def test_metrics_noise_monotonic():
    rng = np.random.default_rng(6)
    img = rng.standard_normal((40, 40, 40)) + 10.0
    mask = brain_mask(img)
    p1 = psnr(img, img + 0.5 * rng.standard_normal(img.shape), mask)
    p2 = psnr(img, img + 2.0 * rng.standard_normal(img.shape), mask)
    assert p1 > p2, (p1, p2)
    s1 = ssim(img, img + 0.5 * rng.standard_normal(img.shape), mask)
    s2 = ssim(img, img + 2.0 * rng.standard_normal(img.shape), mask)
    assert s1 > s2, (s1, s2)


@check("metrics: 形状不一致时抛出 ValueError")
def test_metrics_shape_mismatch():
    a = np.zeros((20, 20, 20))
    b = np.zeros((21, 20, 20))
    mask = np.ones((20, 20, 20), dtype=bool)
    try:
        mse(a, b, mask)
    except ValueError:
        return
    raise AssertionError("形状不一致应抛出 ValueError")


@check("非法参数: r<1 / 未知滤波器 / 非法 axis")
def test_invalid_args():
    vol = np.zeros((10, 10, 10))
    for kwargs in (dict(axis=0, r=0.5), dict(axis=0, r=2, filter_name="bogus"), dict(axis=3, r=2)):
        try:
            degrade_through_plane(vol, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"应抛出 ValueError: {kwargs}")


def main() -> int:
    test_r1_identity()
    test_high_freq_removed_only_on_axis()
    test_sinc_bandlimited_exact()
    test_sinc_odd_length_no_halfshift()
    test_imag_residual_tiny()
    test_shape_spacing_affine()
    test_metrics_identity()
    test_metrics_noise_monotonic()
    test_metrics_shape_mismatch()
    test_invalid_args()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = len(RESULTS) - passed
    print("\n" + "-" * 60)
    print(f"退化/指标单元测试汇总（合成数组）: {passed} passed, {failed} failed, {len(RESULTS)} total")
    if failed:
        for name, ok, err in RESULTS:
            if not ok:
                print(f"  FAILED: {name} -> {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
