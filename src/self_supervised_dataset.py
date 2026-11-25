"""SMORE 自监督训练数据集（Day 3）。

对应论文 Algorithm 1 的样本构造
===============================

设低分辨率 MRI 的 Sinc 插值结果为 g(x,y,z)，其中 x,y 是 in-plane 高分辨率方向，
z 是 through-plane 低分辨率方向。SMORE 的巧思是：

* **训练目标来自 g 自身的高分辨率平面**：g 只在 z 方向损失了分辨率，x-y 平面
  仍是高分辨率。因此可以直接用 g 的 x-y 切片作为"清晰目标"，**无需外部 HR**。
* **训练输入**：沿 x 方向对 x-y 切片施加与 z 方向退化相同形式的一维模糊核
  h(r)，得到 b = h(r) * g。网络学习 b -> g 的映射，即学会"沿一个方向去模糊"。
* **推理**：把学到的 1D 去模糊能力应用到包含 z 方向的 x-z 切片上，恢复 z 方向
  的分辨率。

本项目的轴映射（canonical RAS+，来自 Day 1/2 的确认）
----------------------------------------------------

* 体数据 shape = (R, A, S) = (150, 256, 256)，axis 0=R, 1=A, 2=S。
* through-plane 低分辨率方向 = **R（axis 0）**；in-plane 高分辨率方向 = A、S。
* 因此：SMORE 的 x <-> A，y <-> S，z <-> R。
* 训练切片 = 固定 R 的 A-S 平面 ``g[r,:,:]``，shape (A,S)，沿 axis 0 (=A) 模糊。
* 推理切片 = 固定 S 的 R-A 平面 ``g[:,:,s]``，shape (R,A)，axis 0 (=R) 是待去模糊轴。

训练与推理的模糊方向都是 patch 的 **第 0 个空间轴**，所以同一个 2D 网络可以
从"去 A 方向模糊"迁移到"去 R 方向模糊"（1D 模糊核形式相同）。

严格约束：本模块**只读取 g（Sinc 插值结果）**，绝不读取原始 HR 参考。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_mri
from src.degradation import lowpass_along_axis


def _foreground_scale(g: np.ndarray, q: float = 99.5) -> float:
    """用前景第 q 百分位作为稳健的归一化常数。"""
    fg = g[g > 0]
    if fg.size == 0:
        return 1.0
    s = float(np.percentile(fg, q))
    return s if s > 0 else 1.0


class SMOREDataset(Dataset):
    """从 g 的 in-plane（A-S）切片构造 (模糊输入, 清晰目标) 的 patch 对。"""

    def __init__(
        self,
        g_path: str | Path,
        *,
        r: float = 2.0,
        filter_name: str | list[str] = "gaussian",
        patch_size: int = 32,
        patches_per_slice: int = 32,
        mode: str = "train",
        val_interval: int = 5,
        seed: int = 0,
        scale: float | None = None,
        noise_sigma: float = 0.0,
    ):
        if patch_size <= 0 or patch_size % 2 != 0:
            raise ValueError(f"patch_size 必须是正偶数，得到 {patch_size}")
        if mode not in ("train", "val"):
            raise ValueError(f"mode 必须是 train/val，得到 {mode}")

        vol = load_mri(Path(g_path), canonical=True)
        g = vol.data.astype(np.float64)
        if g.ndim != 3:
            raise ValueError(f"g 必须是三维，得到 {g.shape}")
        self.affine = vol.affine
        self.g_shape = g.shape
        self.r = float(r)
        # 支持单一滤波器（str）或多退化（list），训练样本在多个退化间随机采样
        self.filter_names = [filter_name] if isinstance(filter_name, str) else list(filter_name)
        self.multi = len(self.filter_names) > 1
        self.patch_size = patch_size
        self.patches_per_slice = patches_per_slice
        self.scale = scale if scale is not None else _foreground_scale(g)
        self.mode = mode
        self.seed = seed
        self.noise_sigma = float(noise_sigma)

        # 归一化整个体数据（训练与推理使用同一个 scale）
        g_norm = (g / self.scale).astype(np.float32)

        # 按 R 轴划分 train / val：固定 R 索引，抽取 A-S 切片
        n_r = g.shape[0]
        r_idx = np.arange(n_r)
        val_idx = r_idx[::val_interval] if val_interval > 1 else r_idx[:0]
        train_idx = np.setdiff1d(r_idx, val_idx)

        chosen = train_idx if mode == "train" else val_idx
        self.r_indices = list(int(i) for i in chosen)

        # 预计算 targets（共享）与每个滤波器的 inputs
        self.targets: list[np.ndarray] = []
        self.inputs_by_filter: dict[str, list[np.ndarray]] = {f: [] for f in self.filter_names}
        for i in self.r_indices:
            slc = g_norm[i, :, :]  # (A, S)
            self.targets.append(slc.astype(np.float32))
            for f in self.filter_names:
                blurred = lowpass_along_axis(slc, axis=0, r=self.r, filter_name=f)
                self.inputs_by_filter[f].append(blurred.astype(np.float32))

        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.r_indices) * self.patches_per_slice

    def _random_crop_offset(self, shape: tuple[int, int]) -> tuple[int, int]:
        h, w = shape
        ps = self.patch_size
        if h < ps or w < ps:
            return 0, 0  # 太小则零填充，取左上角
        i0 = int(self._rng.integers(0, h - ps + 1))
        j0 = int(self._rng.integers(0, w - ps + 1))
        return i0, j0

    def _crop_patch(self, arr: np.ndarray, i0: int, j0: int) -> np.ndarray:
        h, w = arr.shape
        ps = self.patch_size
        if h >= ps and w >= ps:
            return arr[i0:i0 + ps, j0:j0 + ps]
        # 太小则零填充（正常数据不会走到这里）
        out = np.zeros((ps, ps), dtype=arr.dtype)
        out[: min(h, ps), : min(w, ps)] = arr[: min(h, ps), : min(w, ps)]
        return out

    def _augment(self, inp: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # 水平/垂直翻转：不改变"模糊沿 axis 0"的方向性质
        if self._rng.random() < 0.5:
            inp = inp[:, ::-1]
            tgt = tgt[:, ::-1]
        if self._rng.random() < 0.5:
            inp = inp[::-1, :]
            tgt = tgt[::-1, :]
        return inp, tgt

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        slice_i = idx // self.patches_per_slice
        slice_i = slice_i % len(self.targets)
        # 多退化时随机选一个滤波器；否则固定唯一滤波器
        fi = int(self._rng.integers(0, len(self.filter_names))) if self.multi else 0
        f = self.filter_names[fi]
        inp_full = self.inputs_by_filter[f][slice_i]
        # 关键：input 与 target 使用同一次随机裁剪位置，保证严格对齐
        i0, j0 = self._random_crop_offset(inp_full.shape)
        inp = self._crop_patch(inp_full, i0, j0)
        tgt = self._crop_patch(self.targets[slice_i], i0, j0)
        inp, tgt = self._augment(inp, tgt)
        if self.noise_sigma > 0:
            # Rician 噪声：在 I/Q 两路加独立高斯噪声后取模
            n1 = self._rng.standard_normal(inp.shape) * self.noise_sigma
            n2 = self._rng.standard_normal(inp.shape) * self.noise_sigma
            inp = np.sqrt((inp + n1) ** 2 + n2 ** 2).astype(np.float32)
        x = torch.from_numpy(inp[None, ...].copy())   # [1,H,W]
        y = torch.from_numpy(tgt[None, ...].copy())   # [1,H,W]
        return x, y


def iter_slices(g_path: str | Path, scale: float | None = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """按顺序遍历 g 的 A-S 切片（用于推理前可视化样本）。"""
    vol = load_mri(Path(g_path), canonical=True)
    g = vol.data.astype(np.float64)
    s = scale if scale is not None else _foreground_scale(g)
    g_norm = (g / s).astype(np.float32)
    for i in range(g.shape[0]):
        slc = g_norm[i, :, :]
        yield slc, lowpass_along_axis(slc, axis=0, r=2.0, filter_name="gaussian")
