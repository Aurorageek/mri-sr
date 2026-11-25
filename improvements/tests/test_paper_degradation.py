"""Paper-aligned degradation (rect × Fermi) 单元测试。

验证 SMORE Eq. (1) 的实现：H(u;r)=rect(ru)·Fermi(u;r)，
Fermi=1/(1+exp((|u|-1/(2r))/(10/L)))。

运行::

    python improvements/tests/test_paper_degradation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.degradation import _centered_freq, _lowpass_mask

RESULTS = []


def check(name):
    def d(fn):
        def w():
            try:
                fn(); RESULTS.append((name, True, "")); print(f"PASS  {name}")
            except Exception as e:  # noqa: BLE001
                RESULTS.append((name, False, repr(e))); print(f"FAIL  {name}\n      {e!r}")
        return w
    return d


@check("paper: 截止频率外严格为 0（rect）")
def test_cutoff():
    for n, r in [(150, 2.0), (256, 2.0), (140, 3.0)]:
        f = _centered_freq(n); fc = 0.5 / r
        m = _lowpass_mask(n, r, "paper")
        assert np.all(m[np.abs(f) > fc + 1e-12] == 0.0)


@check("paper: 截止处 Fermi = 0.5")
def test_fermi_half_at_cutoff():
    n = 256; r = 2.0
    f = _centered_freq(n); fc = 0.5 / r
    m = _lowpass_mask(n, r, "paper")
    idx = np.argmin(np.abs(np.abs(f) - fc))
    assert abs(m[idx] - 0.5) < 0.02, m[idx]


@check("paper: 通带内单调不增、值域 [0,1]")
def test_monotone():
    for n in (150, 256):
        f = _centered_freq(n); m = _lowpass_mask(n, 2.0, "paper")
        order = np.argsort(np.abs(f)); ff = np.abs(f)[order]; mm = m[order]
        inside = ff <= 0.25
        assert np.all(np.diff(mm[inside]) <= 1e-12)
        assert np.all(mm >= 0) and np.all(mm <= 1 + 1e-12)


@check("paper: 过渡带宽随 L 变化（10/L）")
def test_width_scales_with_L():
    # 较小 L -> 更宽过渡 -> 在 0.9*fc 处衰减更多
    def val(n, frac=0.9):
        f = _centered_freq(n); fc = 0.25
        idx = np.argmin(np.abs(np.abs(f) - frac * fc))
        return _lowpass_mask(n, 2.0, "paper")[idx]
    assert val(64) < val(256), (val(64), val(256))


def main() -> int:
    test_cutoff(); test_fermi_half_at_cutoff(); test_monotone(); test_width_scales_with_L()
    passed = sum(1 for _, ok, _ in RESULTS if ok); failed = len(RESULTS) - passed
    print("\n" + "-" * 60)
    print(f"paper 退化测试: {passed} passed, {failed} failed, {len(RESULTS)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
