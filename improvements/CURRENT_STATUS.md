# 当前研究状态

> 更新日期：2026-11-24（Phase 3 结束）

## 当前阶段

Phase 3 完成：已冻结最终方法并完成独立受试者验证。

## 核心结论

1. **退化失配是 Gaussian 特有的**：Fixed-Gaussian SMORE 在 Box/Fermi 上比 Sinc 低
   1.2–1.8 dB；Fixed-Box / Fixed-Fermi 天然鲁棒。
2. **多退化不是必要的**：Uniform-Multi（+0.171）差于 Fixed-Box（+0.284）。
3. **自适应退化估计不可行**：三种退化频谱 corr > 0.99999。
4. **最终方法 Fixed-Box**：开发集 +0.244（worst +0.140），独立测试 +0.244
   （worst +0.062），12/12 为正。

## 当前最佳方法

`SMORE-Fixed-Box`（见 `BEST_METHOD.md`）。

## 已淘汰方案

- Sobel Gradient Loss（冗余）
- 退化对齐训练（失配可忽略）
- 多方向平均融合（冗余）
- Uniform-Multi（差于 Fixed-Box）
- 自适应退化采样（无可估计信号）
- r=6 扩展（r=4 收益已很小）

## 尚未完成 / 下一步

1. 独立测试扩到 8–10 名以获得统计功效。
2. 补充完整 SSIM/梯度指标表与置信区间。
3. 英文论文正文（素材见 `reports/PAPER_EXPERIMENTS.md`）。

## 最近一次 GitHub 更新

- 分支：`research/mri-sr-improvements`
- 地址：https://github.com/Aurorageek/mri-sr/tree/research/mri-sr-improvements
- 本轮报告：`improvements/reports/PHASE3_REPORT.md`
