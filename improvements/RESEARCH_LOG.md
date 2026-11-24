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
