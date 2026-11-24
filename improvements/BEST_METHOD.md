# 当前最佳方法

> 尚无改进方法超越 Baseline；当前最佳 = **SMORE-L1**（作为改进的参考基线）。

## 方法

简化 SMORE(3D) + L1 损失（λ=0）：EDSR 在 in-plane（A-S）平面自监督学习一维去模糊，
推理时在 R-A 平面对 through-plane（R）去模糊。

## 与 Sinc Baseline 的区别

Sinc 是理想带限内插（无学习）；SMORE-L1 用自监督学习恢复被高斯模糊掉的带内细节。

## 当前证据（8 名已用测试受试者，r=2）

- PSNR 30.81 vs Sinc 30.50（+0.32 dB，p=0.008）
- SSIM 0.9338 vs 0.9291
- MAE 64.2 vs 68.8
- Through-plane 梯度误差 1029 vs 1139（-9.7%）
- 3D 梯度误差 811 vs 1019（-20%）

## 复现命令

```powershell
python scripts/train_smore.py --loss l1 --g-path data/processed/<s>/<s>_sinc_r2.nii.gz --experiment-name <s>_l1
python scripts/infer_smore.py --g-path data/processed/<s>/<s>_sinc_r2.nii.gz --checkpoint results/checkpoints/<s>_l1/best.pt --output data/processed/<s>/<s>_smore_l1.nii.gz
```

## 已知局限

- 单方向推理（R-A），无多方向融合。
- 退化轴 = R（本数据矢状位存储），放大倍数受 GT 层厚限制。
- 训练退化与实际退化可能不等价（待研究，方向 A）。
