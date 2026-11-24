"""简化 EDSR 网络（SMORE 用），单通道、无上采样、同尺寸输入输出。

与原论文 / 原始 EDSR 的区别
---------------------------

* **无 BatchNorm**：EDSR 作者指出 BN 会损失像素级范围信息、增大显存，
  且超分任务通常不需要 BN（原论文正是去掉了 BN）。
* **保留 residual scaling**（res_scale）：残差分支乘一个小常数，稳定深网络训练。
* **移除上采样模块**：因为 SMORE 的输入已经是 Sinc 插值到目标数字网格的结果，
  网络的任务是**恢复细节**而非增加像素数量，所以输入输出空间尺寸相同。

本实现是调试起始配置（n_resblocks=8, n_feats=64, res_scale=0.1），
**不等于**原论文的完整网络超参数。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """EDR S残差块：Conv-ReLU-Conv，无 BN，输出加 res_scale * x。"""

    def __init__(self, n_feats: int, res_scale: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale * self.body(x)


class EDSR(nn.Module):
    """简化 EDSR：单通道进、单通道出、同尺寸。"""

    def __init__(
        self,
        n_resblocks: int = 8,
        n_feats: int = 64,
        res_scale: float = 0.1,
        n_channels: int = 1,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.head = nn.Conv2d(n_channels, n_feats, 3, padding=1, bias=True)
        self.body = nn.Sequential(
            *[ResBlock(n_feats, res_scale) for _ in range(n_resblocks)]
        )
        self.tail = nn.Conv2d(n_feats, n_channels, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        # 全局残差：输出 = 输入 + 学到的细节（EDSR 的 long skip）。
        # 对去模糊任务，输入已接近目标，网络只需学习被模糊掉的高频残差。
        return identity + x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
