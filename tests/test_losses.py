"""Day 4 SobelGradientLoss 单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.losses import SobelGradientLoss, combined_loss

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


@check("SobelLoss 输出有限标量，且 pred==target 时为 0")
def test_finite_and_zero():
    s = SobelGradientLoss()
    x = torch.randn(4, 1, 32, 32)
    assert torch.isfinite(s(x, x))
    assert s(x, x).item() == 0.0
    y = torch.randn(4, 1, 32, 32)
    v = s(x, y)
    assert v.ndim == 0 and torch.isfinite(v)


@check("SobelLoss 可反向传播产生梯度")
def test_backward():
    s = SobelGradientLoss()
    x = torch.randn(2, 1, 32, 32, requires_grad=True)
    y = torch.randn(2, 1, 32, 32)
    s(x, y).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@check("lambda=0 时 total 严格等于 L1")
def test_lambda_zero_equals_l1():
    s = SobelGradientLoss()
    x = torch.randn(4, 1, 32, 32)
    y = torch.randn(4, 1, 32, 32)
    total, l1, grad = combined_loss(x, y, s, 0.0)
    assert torch.allclose(total, l1)
    assert torch.allclose(total, F.l1_loss(x, y))
    assert torch.isfinite(grad)  # 仍可计算，用于记录


@check("SobelLoss 的 device/dtype 正确（CPU/float32）")
def test_device_dtype():
    s = SobelGradientLoss()
    x = torch.randn(2, 1, 16, 16)
    y = torch.randn(2, 1, 16, 16)
    v = s(x, y)
    assert v.dtype == torch.float32


@check("SobelLoss 对常数图像输出 0（无伪边缘）")
def test_constant_image():
    s = SobelGradientLoss()
    x = torch.ones(2, 1, 16, 16)
    y = torch.ones(2, 1, 16, 16) * 2.0
    # 常数图，Sobel 梯度都为 0 -> 梯度损失为 0
    assert s(x, y).item() == 0.0


@check("SobelLoss 对高频边缘变化敏感")
def test_sensitive_to_edges():
    s = SobelGradientLoss()
    a = torch.zeros(2, 1, 16, 16)
    b = torch.zeros(2, 1, 16, 16)
    a[:, :, :, 8:] = 1.0          # 垂直阶跃边
    b[:, :, :, 7:] = 1.0          # 边缘偏移 1 像素
    c = torch.zeros(2, 1, 16, 16) # 无边缘
    assert s(a, b).item() > s(a, a).item()  # 有差异
    assert s(a, c).item() > 0.0            # 有边缘 vs 无边缘


def main() -> int:
    test_finite_and_zero()
    test_backward()
    test_lambda_zero_equals_l1()
    test_device_dtype()
    test_constant_image()
    test_sensitive_to_edges()
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = len(RESULTS) - passed
    print("\n" + "-" * 60)
    print(f"Day4 损失单元测试汇总: {passed} passed, {failed} failed, {len(RESULTS)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
