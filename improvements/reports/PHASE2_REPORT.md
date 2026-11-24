# Phase 2 研究报告：退化鲁棒的自监督 MRI 超分辨率

## 1. EXP003–EXP005 的已有结论是否可靠？

部分可靠，但**实验协议需要修正**。核查（EXP006）确认 EXP003–005 属 **Protocol A
（冻结模型跨退化迁移）**：训练数据源始终是 g_gaussian，再用同一 checkpoint 在
Box/Fermi 上推理。这回答的是"固定模型能否跨退化复用"，而非标准 SMORE 的表现。

## 2. 严格的单受试者自监督协议下，多退化训练是否仍然有效？

**是（EXP007）**。在 Protocol B（每种实际退化用其 g 自身训练）下，结果：

| 实际退化 | Sinc | Fixed-G | Multi-Deg |
|---|---:|---:|---:|
| Gaussian | 28.631 | +0.273 | +0.412 |
| Box | 29.993 | -1.637 | +0.001 |
| Fermi | 29.817 | -1.236 | +0.100 |

Fixed-G 在 Box/Fermi 上仍损害重建，Multi-Deg 修复之。Protocol A 结论在 Protocol B 下成立。

## 3. 退化失配导致性能下降的原因有哪些实验证据？

- **残差分析（EXP009）**：Fixed-G 的残差在失配条件**单调有害**（α 越大 PSNR 越低），
  证明其"锐化"放大了振铃/混叠伪影；Through-plane 梯度误差在 Box 上高达 486.8
  （vs Sinc 约 351），即引入了伪边缘。Multi-Deg 残差接近中性（TP-Grad 366.5）。

## 4. 多退化训练是否具有未知退化泛化能力？

**部分泛化（EXP008）**：Multi-Deg{G,B}（不含 Fermi）在未见 Fermi 上 -0.25 dB，
明显好于 Fixed-G 的 -1.24，但差于 Full{G,B,F} 的 +0.10。因此是"相似退化族内的
部分泛化"，不能声称"完全未知退化泛化"。

## 5. 更高退化倍率下的表现如何？

**收益随倍率减小（EXP010）**：r=4 时整体 PSNR 降到 ~22.6，SMORE 增益（Gaussian 上
+0.005）与失配效应（Box 上 -0.13）都大幅缩小。多退化鲁棒性的价值主要在 r=2。

## 6. 是否实现了进一步的自适应改进？

本轮未实现新的自适应方法（如加权采样/条件 EDSR）。保守残差分析（EXP009）表明
α∈[0.5,0.75] 有边际收益（+0.08~+0.13 dB），但低于有意义阈值，未单独作为方法。

## 7. 当前最佳方法

**SMORE-L1 + Multi-Degradation**（多退化自监督训练）：
- 匹配（Gaussian）条件：+0.41 dB（略优于 Fixed-G 的 +0.27）。
- 失配（Box/Fermi）条件：≈0 dB（Fixed-G 为 -1.2~-1.6 dB）。
- 零额外参数量、零额外推理开销。

## 8. 当前方法是否稳定优于 Sinc 和 SMORE Baseline？

- **优于 Sinc**：匹配条件下稳定（+0.41 dB）；失配条件下至少不差于 Sinc（≈0 dB），
  而 Fixed-G 会差于 Sinc。
- **优于 SMORE Baseline（Fixed-G）**：鲁棒性维度上明显更优（失配最差 -0.03 vs -1.64）。

## 9. 是否存在过度平滑或错误高频结构？

Fixed-G 在失配条件存在**错误高频结构（过度锐化/振铃放大）**，已由 TP-Grad 与残差
分析证实。Multi-Deg 未观察到明显错误高频，但在失配条件输出接近 Sinc（即恢复的额外
细节有限，偏保守）。

## 10. 哪些结论具有充分证据，哪些仍属假设？

**有充分证据**：
- 退化失配（训练核≠实际退化）会导致 Fixed-G 过度锐化、差于 Sinc（Protocol A/B、4 受试者一致）。
- 多退化训练修复失配、保持匹配条件性能（EXP004/005/007）。
- 部分未知退化泛化（EXP008）。
- 失配价值主要在 r=2（EXP010）。

**仍属假设/待验证**：
- 4 名受试者均为已用数据，非独立测试集。
- 多退化的机制是"抑制有害锐化"（EXP009 支持，但仅 1 名受试者）。

## 11. 哪些实验需要在新受试者上继续验证？

在**完全独立的 8–10 名新测试受试者**上，验证：
- 多退化训练在 r=2 下相对 Fixed-G 的鲁棒性优势（平均失配 ΔPSNR）。
- 匹配条件的 PSNR/SSIM 是否持平或略升。

## 12. 当前结果是否具备形成课程论文主线的潜力？

**具备**。可形成完整的论证链：

- **Research Problem**：自监督 MRI SR 对退化模型假设敏感，失配时反而损害重建。
- **Observation**：Fixed-G SMORE 在 Box/Fermi 上差于 Sinc（-1.2~-1.6 dB）。
- **Hypothesis**：多退化训练提升鲁棒性。
- **Method**：自监督训练时随机采样 {Gaussian,Box,Fermi}。
- **Evidence**：Protocol A/B、4 受试者、leave-one-out、残差机制分析。
- **Conclusion**：多退化训练以零额外开销显著提升鲁棒性，但属于"抑制有害锐化"而非"恢复额外细节"。

## 13. 最新 Git commit 和 GitHub 分支

- 分支：`research/mri-sr-improvements`
- 见 git log；访问：https://github.com/Aurorageek/mri-sr/tree/research/mri-sr-improvements
