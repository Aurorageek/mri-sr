"""梯度评价指标回归测试（Day 6 修正）。

验证 through-plane / 3D 梯度误差的方向性，防止 scipy.ndimage.sobel 默认 axis
导致的错误再次出现。运行::

    python improvements/tests/test_gradient_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import (
    gradient_3d_mae,
    gradient_mae,
    through_plane_gradient_mae,
)

RESULTS: list[tuple[str, bool, str]] = []


def check(name):
    def d(fn):
        def w():
            try:
                fn(); RESULTS.append((name, True, "")); print(f"PASS  {name}")
            except Exception as e:  # noqa: BLE001
                RESULTS.append((name, False, repr(e))); print(f"FAIL  {name}\n      {e!r}")
        return w
    return d


def _ramp(shape, axis):
    ref = np.zeros(shape)
    sl = [slice(None)] * 3
    for i in range(shape[axis]):
        sl[axis] = i
        ref[tuple(sl)] = i * 10.0
    return ref


@check("through_plane 梯度误差只对齐指定轴")
def test_through_plane_directionality():
    shape = (32, 32, 32)
    mask = np.ones(shape, dtype=bool)
    zero = np.zeros(shape)
    for ax in (0, 1, 2):
        ref = _ramp(shape, ax)
        for k in (0, 1, 2):
            v = through_plane_gradient_mae(ref, zero, mask, axis=k, spacing=1.0)
            assert (abs(v) > 1.0) == (k == ax), f"axis={ax}, metric_axis={k}, v={v}"


@check("3D 梯度误差对所有轴敏感")
def test_3d_directionality():
    shape = (32, 32, 32)
    mask = np.ones(shape, dtype=bool)
    zero = np.zeros(shape)
    for ax in (0, 1, 2):
        v = gradient_3d_mae(_ramp(shape, ax), zero, mask, spacing=(1.0, 1.0, 1.0))
        assert v > 1.0, v


@check("legacy gradient_mae 仅对 S 轴（axis=-1）敏感")
def test_legacy_is_s_axis():
    shape = (32, 32, 32)
    mask = np.ones(shape, dtype=bool)
    zero = np.zeros(shape)
    assert gradient_mae(_ramp(shape, 2), zero, mask) > 1.0
    assert gradient_mae(_ramp(shape, 0), zero, mask) == 0.0
    assert gradient_mae(_ramp(shape, 1), zero, mask) == 0.0


@check("voxel spacing 会按轴加权 3D 梯度")
def test_3d_spacing_weighting():
    shape = (32, 32, 32)
    mask = np.ones(shape, dtype=bool)
    zero = np.zeros(shape)
    ref = _ramp(shape, 0)  # 沿 axis0 变化
    # axis0 spacing 大 -> 除以更大 spacing -> 梯度误差更小
    v_small = gradient_3d_mae(ref, zero, mask, spacing=(2.0, 1.0, 1.0))
    v_large = gradient_3d_mae(ref, zero, mask, spacing=(1.0, 1.0, 1.0))
    assert v_small < v_large, (v_small, v_large)


def main() -> int:
    test_through_plane_directionality()
    test_3d_directionality()
    test_legacy_is_s_axis()
    test_3d_spacing_weighting()
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = len(RESULTS) - passed
    print("\n" + "-" * 60)
    print(f"梯度指标回归测试: {passed} passed, {failed} failed, {len(RESULTS)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
