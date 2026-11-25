# 当前研究状态

> 更新日期：2026-11-24（Phase 4 结束）

## 当前阶段

Phase 4 完成：原论文对齐基线 + 跨平面可靠性假设验证 + 备选噪声鲁棒方向。

## 核心结论

1. **原论文对齐退化很重要**：rect×Fermi 下 SMORE 单方向增益 +0.78~1.02 dB，
   远大于 Gaussian 模拟的 +0.24~0.41 dB。
2. **跨平面分歧**：能反映误差大小（ρ≈0.30–0.35，偏相关 0.22–0.26），
   **不能**识别有害修正（ρ≈0–0.1）。
3. **RG-SMORE 失败（DISCARD）**：劣于 Average/FBA 与能量匹配固定缩放；
   门控削弱了残差与真实残差的对齐。
4. **M=2 平均/FBA 仅略优于单方向**（+0.003~0.016 dB），但稳定。
5. **噪声鲁棒（KEEP）**：干净训练 SMORE 在噪声下低于 Sinc；噪声增强训练
   （σ=0.02/0.05）以近零/小代价修复。

## 当前最佳方法

Paper-Aligned SMORE + M=2 平均/FBA（+ 可选噪声增强）。见 `BEST_METHOD.md`。

## 已淘汰方案

Sobel loss、退化对齐、多方向平均（Phase 2 结论）、Uniform-Multi、自适应采样、
Fixed-Box（协议变更后）、**RG-SMORE**。

## 下一步

1. 噪声鲁棒方法在 8–10 名新受试者上验证（当前 2 名）。
2. 完整指标表（SSIM/梯度）与统计检验。
3. 英文论文正文；正式写作前补查"分歧引导 SR"相关文献。

## 最近 GitHub

- 分支 `research/mri-sr-improvements`；报告 `PAPER_EXPERIMENTS.md`、
  `PHASE4_REPORT.md`。
