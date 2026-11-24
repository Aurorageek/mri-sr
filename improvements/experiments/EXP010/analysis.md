# EXP010：高倍率退化（r=4）验证

- **Research Question**：多退化鲁棒性在高倍率下是否仍成立？
- **Method**：IXI002，r=4，三种实际退化 × Fixed-G / Multi-Deg（Protocol B）。
- **Results（ΔPSNR vs Sinc）**：

| 实际退化 | Sinc | Fixed-G | Multi-Deg |
|---|---:|---:|---:|
| Gaussian | 22.579 | +0.005 | +0.002 |
| Box | 22.679 | -0.131 | -0.033 |
| Fermi | 22.710 | -0.087 | -0.025 |

- **Conclusion**：r=4 时**整体 PSNR 大幅下降（~22.6 vs r=2 的 ~29）**，且
  （1）SMORE 的增益几乎消失（Gaussian 上仅 +0.005），（2）退化失配效应也大幅缩小
  （Fixed-G 在 Box 上 -0.13 vs r=2 的 -1.64）。
  原因：r=4 截止频率极低（f_cut=0.125），可恢复细节很少，且各滤波器在如此低的
  截止下产生的 LR 几乎相同。
- **Decision**：多退化鲁棒性的价值主要在**中等退化倍率（r=2）**；r=4 时问题过难、
  失配可忽略，无需再扩展 r=6。
