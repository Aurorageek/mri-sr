# 研究日志

## 2026-11-24 — 基础审查与评价指标修正（Phase 1）

- **研究问题**：Day 6 报告的 Gradient MAE 指标是否真的衡量了 through-plane 结构误差？
- **发现**：`scipy.ndimage.sobel(vol)` 默认 `axis=-1`，只计算 S 轴 Sobel，而非 3D
  梯度幅值；且未对准 through-plane（R=axis0）方向。→ 确认评价指标错误。
- **修改**：`src/metrics.py` 新增 `through_plane_gradient_mae`（R 轴，含 spacing）、
  `gradient_3d_mae`（三轴 spacing 加权）；旧 `gradient_mae` 保留为 legacy（S 轴）。
- **验证**：用斜坡数组确认方向性；用 8 受试者现有重建结果重算指标。
- **结果**：SMORE-L1 vs Sinc 的 through-plane 误差 -9.7%、3D 梯度误差 -20%（p=0.008），
  结论方向不变（SMORE > Sinc）。Grad 相对 L1 仍为噪声级。
- **决策**：指标修正成立；Sobel Gradient Loss 维持 DISCARD。
- **下一步**：方向 A 退化算子对齐的数值分析。

## 2026-11-24 — EXP001 方向 A：退化算子对齐（数值分析，无需训练）

- **研究问题**：训练退化与推理实际退化是否等价？
- **结果**：真实 MRI 上 D_train≈D_real（corr 0.99–0.999，高频残留<0.5%）；含高频合成信号才显差异（corr 0.87）。
- **结论**：对真实 MRI，退化失配可忽略；假设被证伪。
- **决策**：DISCARD 方向 A 的训练实验（负结果，节省算力）。
- **下一步**：方向 D（多方向重建融合，复用 checkpoint、无需训练）。

## 2026-11-24 — EXP002 方向 D：多方向重建融合（复用 checkpoint）

- **研究问题**：R-A 与 R-S 双方向推理融合能否降低误差？
- **结果**：single_RA=28.904、single_RS=28.908、fused=28.906 dB，几乎一致。
- **结论**：两方向高度冗余（都沿 R 去模糊），简单平均无增益。
- **决策**：DISCARD（负结果，单方向 R-A 已足够）。

## 2026-11-24 — EXP003 方向 B：退化失配零训练预实验（重要发现 KEEP）

- **研究问题**：Gaussian 训练的 SMORE 在 Box/Fermi 生成的 LR 上是否仍有效？
- **结果**：Gaussian 条件 SMORE 优于 Sinc（+0.27 dB）；但 Box 条件 **-1.63 dB**、
  Fermi 条件 **-1.24 dB**，即失配时 SMORE 反而差于 Sinc。
- **结论**：**退化失配现象显著**，模型对滤波器敏感，失配时主动损害重建。
- **决策**：KEEP，直接支持方向 B（失配矩阵）与 C（多退化鲁棒训练）。
- **下一步**：实现多退化训练（Direction C），验证能否改善失配鲁棒性。

## 2026-11-24 — EXP004 方向 C：多退化鲁棒训练（正结果 KEEP）

- **研究问题**：多退化训练能否修复失配？
- **方法**：SMOREDataset 支持多退化（Gaussian/Box/Fermi 随机采样），训练 IXI002_multideg。
- **结果**：Box 条件从 -1.63 dB → ≈0，Fermi 从 -1.24 → +0.10，Gaussian 从 +0.27 → +0.41。
- **结论**：多退化训练修复失配，是有效且低成本的鲁棒性改进。
- **决策**：KEEP；下一步在更多受试者验证。
