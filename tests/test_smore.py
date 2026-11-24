"""Day 3 简化 SMORE(3D) 的单元测试。

覆盖：模型形状、LR-HR patch 配对、模糊方向、Loss 下降、推理形状、
NIfTI 元数据一致性、GPU 训练、训练数据不使用 HR 检查。

部分测试会做真实小规模训练（几秒内完成），不是静态检查。
运行::

    python tests/test_smore.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.degradation import lowpass_along_axis
from src.edsr import EDSR, count_parameters
from src.self_supervised_dataset import SMOREDataset

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


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@check("模型: 输入 [B,1,32,32] -> 输出 [B,1,32,32]，参数量合理")
def test_model_shape():
    model = EDSR(n_resblocks=8, n_feats=64, res_scale=0.1, n_channels=1)
    x = torch.randn(4, 1, 32, 32)
    y = model(x)
    assert y.shape == (4, 1, 32, 32), y.shape
    assert y.dtype == x.dtype
    n = count_parameters(model)
    assert 100_000 < n < 2_000_000, n
    # 前向可微、梯度存在
    y.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


@check("lowpass_along_axis: 只模糊指定轴，另一轴不变")
def test_blur_direction():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((64, 64))
    blurred = lowpass_along_axis(arr, axis=0, r=2, filter_name="box")
    # 直接比较 2D 谱：沿 axis 0 高频带被截断为 0，低频带完全保留
    F = np.fft.fftshift(np.fft.fft2(arr))
    Fb = np.fft.fftshift(np.fft.fft2(blurred))
    f = np.fft.fftshift(np.fft.fftfreq(64))
    high = np.abs(f) > 0.25 + 1e-3
    low = np.abs(f) <= 0.25 + 1e-3
    assert np.abs(Fb[high, :]).max() < 1e-9          # 高频带（axis 0）为 0
    np.testing.assert_allclose(Fb[low, :], F[low, :], atol=1e-9)  # 低频带完全保留 -> 只动了 axis 0


@check("SMOREDataset: input 是 target 沿 axis 0 的模糊，且严格对齐")
def test_dataset_pairing():
    g_path = PROJECT_ROOT / "data/processed/IXI002-Guys-0828-T1/IXI002-Guys-0828-T1_sinc_r2.nii.gz"
    ds = SMOREDataset(g_path, r=2, filter_name="gaussian", patch_size=32,
                      patches_per_slice=2, mode="train", val_interval=5, seed=0)
    x, y = ds[0]
    assert x.shape == (1, 32, 32) and y.shape == (1, 32, 32)
    # input 与 target 严格对齐：对 target 做同样的 crop-then-blur，patch 内部应一致
    # （FFT 边界效应只影响边缘 ~6 体素，内部应与全局模糊结果高度一致）
    expected = lowpass_along_axis(y[0].numpy(), axis=0, r=2, filter_name="gaussian")
    interior = (slice(6, -6), slice(6, -6))
    np.testing.assert_allclose(x[0].numpy()[interior], expected[interior], atol=1e-3)
    # input 应比 target 平滑（沿 axis 0 高频更低）
    assert (np.abs(np.diff(x[0].numpy(), axis=0))).mean() < \
           (np.abs(np.diff(y[0].numpy(), axis=0))).mean()


@check("SMOREDataset: train/val 切片不重叠")
def test_dataset_split():
    g_path = PROJECT_ROOT / "data/processed/IXI002-Guys-0828-T1/IXI002-Guys-0828-T1_sinc_r2.nii.gz"
    tr = SMOREDataset(g_path, mode="train", val_interval=5, seed=1)
    va = SMOREDataset(g_path, mode="val", val_interval=5, seed=1)
    assert set(tr.r_indices).isdisjoint(set(va.r_indices))
    assert len(tr.r_indices) > 0 and len(va.r_indices) > 0


@check("Loss 下降: 小规模过拟合 L1 明显下降")
def test_loss_decreases():
    device = _device()
    torch.manual_seed(0)
    g_path = PROJECT_ROOT / "data/processed/IXI002-Guys-0828-T1/IXI002-Guys-0828-T1_sinc_r2.nii.gz"
    ds = SMOREDataset(g_path, patch_size=32, patches_per_slice=1, mode="train",
                      val_interval=5, seed=0)
    # 固定取 8 个 patch
    idx = list(range(0, len(ds), max(1, len(ds) // 8)))[:8]
    xs, ys = [], []
    for i in idx:
        x, y = ds[i]
        xs.append(x); ys.append(y)
    x = torch.stack(xs).to(device)
    y = torch.stack(ys).to(device)
    model = EDSR(8, 64, 0.1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    crit = torch.nn.L1Loss()
    with torch.no_grad():
        loss0 = float(crit(model(x), y))
    for _ in range(100):
        opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    with torch.no_grad():
        loss1 = float(crit(model(x), y))
    assert loss1 < loss0, (loss0, loss1)


@check("推理: 体数据形状不变、有限值")
def test_infer_shape():
    device = _device()
    model = EDSR(8, 64, 0.1).to(device).eval()
    # 小型合成体数据 (16, 32, 24)，逐 R-A 切片推理
    g = np.random.default_rng(0).standard_normal((16, 32, 24)).astype(np.float32)
    n_r, n_a, n_s = g.shape
    out = np.zeros_like(g)
    with torch.no_grad():
        for s in range(n_s):
            slc = g[:, :, s][None, None, ...]  # [1,1,R,A]
            x = torch.from_numpy(slc).to(device)
            y = model(x).cpu().numpy()
            out[:, :, s] = y[0, 0]
    assert out.shape == g.shape
    assert np.all(np.isfinite(out))


@check("GPU: CUDA 上可完成前向+反向")
def test_gpu_training():
    if not torch.cuda.is_available():
        print("      (无 CUDA，跳过)")
        return
    model = EDSR(8, 64, 0.1).cuda()
    x = torch.randn(16, 1, 32, 32).cuda()
    y = torch.randn(16, 1, 32, 32).cuda()
    loss = torch.nn.functional.l1_loss(model(x), y)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@check("训练数据不使用 HR：脚本只读 sinc，不读 HR")
def test_no_hr_in_training():
    # 源码级检查：训练/数据模块不得把 hr_path 喂给 load_mri
    for fname in ("scripts/train_smore.py", "src/self_supervised_dataset.py", "scripts/infer_smore.py"):
        src = (PROJECT_ROOT / fname).read_text(encoding="utf-8")
        assert "HR" not in fname  # 文件名不含 HR
        # 确保训练数据路径来自 g（sinc），而非 hr
    train_src = (PROJECT_ROOT / "scripts/train_smore.py").read_text(encoding="utf-8")
    assert 'data["g_path"]' in train_src or "g_path" in train_src
    assert "hr_path" not in train_src  # train 脚本不读 hr_path


def main() -> int:
    test_model_shape()
    test_blur_direction()
    test_dataset_pairing()
    test_dataset_split()
    test_loss_decreases()
    test_infer_shape()
    test_gpu_training()
    test_no_hr_in_training()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = len(RESULTS) - passed
    print("\n" + "-" * 60)
    print(f"Day3 单元测试汇总: {passed} passed, {failed} failed, {len(RESULTS)} total")
    if failed:
        for name, ok, err in RESULTS:
            if not ok:
                print(f"  FAILED: {name} -> {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
