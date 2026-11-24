# 当前研究状态

> 更新日期：2026-11-24

## 当前阶段

第三阶段完成：已发现并验证一个具有研究价值的改进（多退化鲁棒训练）。

## 主要研究发现

1. **评价指标 bug**：`scipy.ndimage.sobel` 默认只算 S 轴，已修正为正确的
   through-plane（R）与 3D 梯度误差（含 spacing）。
2. **退化对齐（EXP001）**：训练退化 ≈ 实际退化（真实 MRI corr>0.99），无需对齐训练。
3. **多方向融合（EXP002）**：R-A 与 R-S 高度冗余，融合无增益。
4. **退化失配（EXP003）**：固定 Gaussian 模型在 Box/Fermi 上损害重建
   （最差 -1.6~-2.4 dB，低于 Sinc）。
5. **多退化训练（EXP004/005）**：从 {Gaussian,Box,Fermi} 随机采样退化核训练，
   **跨 4 名受试者一致修复失配**（最差 Δ 从 ≈-2.0 dB 到 ≈0 dB），匹配条件持平或略升。

## 当前最佳方法

**SMORE-L1 + Multi-Degradation**（见 `BEST_METHOD.md`）。

## 已淘汰方案

- Sobel Gradient Loss（与 L1 冗余）
- 退化对齐训练（失配可忽略）
- 简单多方向平均融合（冗余）

## 尚未解决 / 可继续

- r=4 倍率、更多退化类型（噪声、不同截止频率）的鲁棒性
- 在完全独立的未用测试集上做最终验证
- 退化参数未知时的"盲超分"设定（不依赖已知退化标签）

## 最近一次 GitHub 更新

- 分支：`research/mri-sr-improvements`，commit `a67f641`（EXP005 验证）。
- 访问：https://github.com/Aurorageek/mri-sr/tree/research/mri-sr-improvements
