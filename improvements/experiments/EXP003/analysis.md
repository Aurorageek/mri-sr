# EXP003：退化模型失配的零训练预实验（重要发现）

- **Research Question**：Gaussian 训练的 SMORE 在 Box/Fermi 生成的 LR 上是否仍然有效？
- **Hypothesis**：训练退化与实际退化失配会导致重建下降，甚至低于 Sinc。
- **Method**：复用 day4_l1（Gaussian 训练）checkpoint，在 IXI002 上用 Gaussian/Box/Fermi
  三种滤波器生成 LR，分别推理并评价。
- **Results（r=2，IXI002）**：

| 实际退化 | Sinc PSNR | SMORE(Gaussian) PSNR | Δ(SMORE-Sinc) |
|---|---:|---:|---:|
| Gaussian | 28.631 | 28.904 | **+0.27** |
| Box | 29.993 | 28.360 | **-1.63** |
| Fermi | 29.817 | 28.573 | **-1.24** |

- **Conclusion**：**退化失配现象显著**。Gaussian 训练模型只在 Gaussian 条件获益；
  在 Box/Fermi 条件下不仅无益，反而明显差于 Sinc（-1.2~-1.6 dB）。
- **Decision**：**KEEP**——这是有研究价值的问题，直接支持方向 B/C。
- **解释（假设）**：模型学到"高斯模糊→锐化"的映射；对 Box/Fermi 的振铃/混叠
  过度锐化，放大了伪影。待进一步验证。
- **代码**：`improvements/scripts/degradation_robustness_pilot.py`。
