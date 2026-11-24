# EXP012：完整单退化基线矩阵（关键发现：失配是 Gaussian 特有）

- **Research Question**：与所有单退化方法相比，Uniform-Multi 是否真的更优？是否存在更简单的单退化方法？
- **Method**：补齐 Fixed-Box、Fixed-Fermi 在 3 种实际退化上的自监督训练（Protocol B），
  构建完整 3×5 矩阵。
- **Results（IXI002，r=2，ΔPSNR vs Sinc）**：

| actual | Sinc | Fixed-G | Fixed-Box | Fixed-Fermi | Multi |
|---|---:|---:|---:|---:|---:|
| gaussian | 28.631 | +0.273 | +0.222 | +0.224 | +0.412 |
| box | 29.993 | **-1.637** | +0.340 | +0.192 | +0.001 |
| fermi | 29.817 | **-1.236** | +0.291 | +0.248 | +0.100 |
| mean | — | -0.867 | **+0.284** | +0.221 | +0.171 |
| worst | — | -1.637 | **+0.222** | +0.192 | +0.001 |

- **Conclusion（关键修正）**：
  1. **退化失配是 Gaussian 特有**：只有 Fixed-Gaussian 在 Box/Fermi 上大幅下降（-1.6/-1.2 dB）。
  2. **Fixed-Box 与 Fixed-Fermi 天然鲁棒**：跨所有条件最差 +0.19~+0.22 dB，无需多退化。
  3. **Uniform-Multi 反而更差**（mean +0.17 < Fixed-Box +0.28），因为它学了"折中"策略。
  4. 原因（假设）：Gaussian 是平滑滤波，训练的锐化在 Box 的硬截断上过冲；Box/Fermi
     是"更锐利"的核，其去模糊更易迁移。
- **Decision**：**修订方法方向**——最终方法应为 **SMORE-Fixed-Box（或 Fixed-Fermi）**，
  而非多退化。多退化不再作为最终方法。
- **下一步**：在更多受试者验证 Fixed-Box 的稳健性。
