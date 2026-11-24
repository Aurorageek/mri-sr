# EXP007：Protocol B 正式验证退化失配（核心复核）

- **Research Question**：在正确的单受试者自监督适配协议（Protocol B）下，退化失配是否仍导致性能下降？多退化训练是否仍有效？
- **Method**：对每种实际退化 D_actual ∈ {Gaussian, Box, Fermi}，用该 g 自身训练
  Fixed-Gaussian 与 Multi-Deg 模型，在同一 g 上推理评价（IXI002，r=2）。
- **Results（ΔPSNR vs Sinc）**：

| 实际退化 | Sinc | Fixed-Gaussian（自适配） | Multi-Deg（自适配） |
|---|---:|---:|---:|
| Gaussian | 28.631 | +0.273 | +0.412 |
| Box | 29.993 | **-1.637** | +0.001 |
| Fermi | 29.817 | **-1.236** | +0.100 |

- **Through-plane 梯度误差（Box 条件）**：Fixed-G=486.8（远差于 Sinc 约 351，
  即引入伪边缘），Multi-D=366.5（接近 Sinc，无伪边缘）。
- **Conclusion**：**Protocol A 结论在 Protocol B 下依然成立**。失配的本质是
  **训练模糊核（Gaussian）≠ 实际退化（Box/Fermi）**，与训练数据来自哪份 g 无关。
  Fixed-Gaussian 在 Box/Fermi 上过度锐化振铃/混叠，反而差于 Sinc；多退化训练修复之。
- **Decision**：**KEEP（强）**。退化失配现象与多退化修复均经严格协议确认。
- **诚实说明**：Multi-Deg 在 Box 上 ≈Sinc（+0.001），即"不再有害"而非"恢复更多细节"；
  在 Gaussian 上真正改善（+0.41）、Fermi 上略改善（+0.10）。
