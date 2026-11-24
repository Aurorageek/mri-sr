# EXP008：Leave-One-Degradation-Out（未知退化泛化，部分泛化）

- **Research Question**：Multi-Deg 训练不含某退化时，对该未见退化是否仍鲁棒？
- **Method**：训练 Multi-Deg{Gaussian, Box}（不含 Fermi）于 g_fermi 上（Protocol B），
  在 Fermi 上测试，与 Fixed-G、Full Multi-Deg{G,B,F} 比较。
- **Results（实际退化=Fermi，Sinc PSNR=29.817，ΔPSNR）**：

| 方法 | PSNR | Δ vs Sinc |
|---|---:|---:|
| Fixed-Gaussian（G） | 28.580 | **-1.24** |
| Multi-Deg{G,B}（leave-out F） | 29.570 | **-0.25** |
| Multi-Deg{G,B,F}（full） | 29.916 | **+0.10** |

- **Conclusion**：多退化训练有**部分未知退化泛化能力**（见 G+B 使 F 从 -1.24 改善到
  -0.25），但**完整覆盖（含 F）仍最优**（+0.10）。因此不能声称"完全未知退化泛化"，
  只能声称"在相似退化族内具有部分泛化"。
- **Decision**：KEEP（对鲁棒性边界的刻画）。
