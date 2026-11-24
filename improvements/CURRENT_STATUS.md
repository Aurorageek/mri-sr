# 当前研究状态

> 更新日期：2026-11-24（第二轮自主科研结束）

## 当前阶段

第二轮研究完成（Phase 2–4）：退化鲁棒方向已形成完整论证链。

## 核心结论

**多退化自监督训练（Multi-Degradation SMORE）** 是当前最有价值的改进：

1. 失配问题真实存在：Fixed-Gaussian SMORE 在 Box/Fermi 退化上差于 Sinc（-1.2~-1.6 dB），
   Protocol A 与 Protocol B 均已确认（EXP003/007，4 受试者）。
2. 多退化训练修复失配（失配最差 Δ ≈0 dB），且匹配条件略升（+0.41 dB）。
3. 机制：多退化靠"抑制有害锐化"（残差中性）而非"恢复额外细节"（EXP009）。
4. 部分未知退化泛化（EXP008）；价值主要在 r=2（EXP010）。

## 当前最佳方法

`SMORE-L1 + Multi-Degradation`（见 `BEST_METHOD.md`）。

## 已淘汰方案

- Sobel Gradient Loss（冗余）
- 退化对齐训练（失配可忽略）
- 简单多方向平均融合（冗余）
- r=6 高倍率扩展（r=4 已显示收益锐减）

## 尚未完成 / 下一步

1. **在完全独立的 8–10 名新测试受试者上做最终验证**（当前 4 名均为已用数据）。
2. 若需更强改进，可探索退化条件调制或自适应退化采样（但残差分析显示边际收益有限）。

## 最近一次 GitHub 更新

- 分支：`research/mri-sr-improvements`。
- 访问：https://github.com/Aurorageek/mri-sr/tree/research/mri-sr-improvements
