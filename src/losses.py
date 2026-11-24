"""损失函数（Day 4）：L1 像素损失 + Sobel 梯度结构损失。

数学定义
--------

总损失：:

    L_total = L1(pred, target) + lambda_grad * L_grad(pred, target)

其中梯度损失为两个方向 Sobel 边缘图的平均绝对误差：:

    L_grad = mean( |Sobel_x(pred) - Sobel_x(target)|
                 + |Sobel_y(pred) - Sobel_y(target)| )

Sobel 核是固定的（register_buffer，不参与训练更新），但卷积操作本身可微，
所以 L_grad 对网络参数有梯度，能反向传播。

设计说明
--------

* 灰度 MRI 单通道，Sobel 核为 1x1x3x3。
* 用 ``reflect`` padding 再做无 padding 卷积，避免边界补零产生的人为边缘
  主导损失。
* 核、pred、target 的 device/dtype 自动对齐（buffer 随模块移动）。
* ``lambda_grad=0`` 时总损失严格等于 L1（梯度项仍可计算，用于记录，但不加入）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


SOBEL_X = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
SOBEL_Y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]


class SobelGradientLoss(nn.Module):
    """Sobel 边缘一致性损失（单通道灰度）。"""

    def __init__(self):
        super().__init__()
        kx = torch.tensor(SOBEL_X, dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor(SOBEL_Y, dtype=torch.float32).view(1, 1, 3, 3)
        # register_buffer：随模型移动到正确 device，但不参与梯度更新
        self.register_buffer("kx", kx, persistent=False)
        self.register_buffer("ky", ky, persistent=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"pred 与 target 形状不一致: {pred.shape} vs {target.shape}")
        # reflect padding 避免边界补零伪影
        p = F.pad(pred, (1, 1, 1, 1), mode="reflect")
        t = F.pad(target, (1, 1, 1, 1), mode="reflect")
        gx_p = F.conv2d(p, self.kx)
        gy_p = F.conv2d(p, self.ky)
        gx_t = F.conv2d(t, self.kx)
        gy_t = F.conv2d(t, self.ky)
        return torch.mean(torch.abs(gx_p - gx_t) + torch.abs(gy_p - gy_t))


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sobel: SobelGradientLoss,
    lambda_grad: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (total, l1, grad)。

    total 是真正用于反向传播的损失；l1 与 grad 是未加权/未组合的分项，
    用于日志记录与跨实验比较（不同 lambda 的 total 定义不同，不能直接比）。
    """
    l1 = F.l1_loss(pred, target)
    grad = sobel(pred, target)
    total = l1 + lambda_grad * grad
    return total, l1, grad
