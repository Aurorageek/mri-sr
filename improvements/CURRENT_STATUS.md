# 当前研究状态

> 更新日期：2026-11-24

## 当前阶段

第二阶段：发现问题（方向 A 退化算子对齐的数值分析）

## 已完成

- [x] 项目审查（代码、Day 6 报告、结果、Git 历史）
- [x] 修复评价指标：`scipy.ndimage.sobel` 默认 axis=-1 只算 S 轴 → 新增正确的
      through-plane 梯度误差（R 轴）与 3D 梯度误差（含 voxel spacing）
- [x] 用斜坡数组验证梯度方向正确性
- [x] 用现有重建结果重算指标（无需重训）：SMORE-L1 vs Sinc 的 through-plane
      误差 -9.7%、3D 梯度误差 -20%（p=0.008）
- [x] 冻结 Day 1–6 基线（main commit `8b2f42c`）
- [x] 建立 improvements/ 研究目录与协议

## 主要发现

1. SMORE > Sinc 稳健（PSNR +0.32 dB，结构误差显著下降）。
2. Sobel Gradient Loss ≈ 无效（与 L1 冗余），已 DISCARD。
3. 训练退化（单轴低通）与推理实际退化（低通+下采样+Sinc）在频域可能不等价——
   待数值验证。

## 下一轮

- 方向 A 数值分析：合成信号 + 真实切片比较两种退化的频率响应与采样相位。

## 最近一次 GitHub 更新

- 待提交：梯度指标修正 + improvements/ 初始化（research 分支）。
