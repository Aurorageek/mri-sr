# 实验协议与冻结基线

## 1. 冻结基线（Day 1–6）

- **Git commit（main 分支）**：`8b2f42c`（day6 review）。研究分支：
  `research/mri-sr-improvements`。
- **数据**：10 例 IXI T1。开发集 = IXI002、IXI012；已用测试集 = 8 例
  （IXI013/014/015/016/017/019/020/021）。
- **退化**：r=2，through-plane 轴 = **R（canonical axis 0）**，gaussian 抗混叠
  低通 + 下采样 + Sinc 插值。voxel spacing = (1.2, 0.9375, 0.9375) mm。
- **网络**：简化 EDSR（8 残差块 / 64 通道 / res_scale 0.1 / 无 BN / 无上采样 /
  全局残差），单通道。
- **自监督训练**：A-S in-plane 切片，沿 A（切片 axis 0）模糊 → 清晰目标。
- **训练预算**：Adam lr=1e-4，batch 16，3000 步，seed 42，FP32，num_workers=0。
- **方法**：Sinc、SMORE-L1（λ=0）、SMORE-Grad（λ=0.10 Sobel）。

## 2. 评价指标（Day 6 修正后）

| 指标 | 定义 | 状态 |
|---|---|---|
| PSNR | 10·log10(range²/MSE)，range = HR 前景动态范围 | 正确 |
| SSIM | 3D 窗 win_size=7，mask 内 | 正确 |
| MAE | `mean |ref-test|` | Day 6 修正（此前误用 RMSE） |
| Through-plane Gradient Error | `sobel(ref, axis=0)/1.2 - sobel(test, axis=0)/1.2` 的 MAE | **Day 6 修正**（此前误用 S 轴） |
| 3D Gradient Error | 三轴 `sobel/space` 合成的梯度幅值 MAE | **Day 6 修正**（新增） |
| 历史遗留 Gradient MAE | `sobel(·, axis=-1)`（仅 S 轴） | 保留对照，标注为 legacy |

- 评价 mask：前景 mask（HR > 0.01·P99.9，3×3×3 腐蚀×2），**非严格脑分割**。
- 强度域：原始强度；data_range 由 HR 前景决定，所有方法统一。

## 3. 数据隔离

- 自监督训练只用 `_sinc_r2.nii.gz`（g），HR 只用于最终评价。
- 已用测试集（8 例）已被查看分析，**不再作为完全独立的最终测试集**。
- 后续探索优先用开发集（IXI002/IXI012）；若需要新测试集，应获取新受试者并在
  协议冻结前不查看其 HR 结果。

## 4. 方法筛选原则

- 初筛参考：相对 SMORE-L1 平均 PSNR 提升约 +0.1 dB 作为继续研究的弱参考，但不
  机械套用；结合 SSIM、结构误差、逐受试者、最差情况与开销综合判断。
- 不事后修改主要指标以让实验"成功"。

## 5. 已知评价修正记录

1. **MAE→RMSE**：Day 4/5 曾把 `mse**0.5` 标为 MAE，Day 6 已修。
2. **Gradient MAE 轴错误**：Day 4/5 用 `ndimage.sobel`（默认 axis=-1=S 轴），
   既非 3D 也非 through-plane（R）。Day 6 已新增正确的 through-plane 与 3D 梯度
   误差，旧值保留为 legacy。修正后 SMORE-L1 vs Sinc：through-plane 误差 -9.7%、
   3D 梯度误差 -20%，结论方向不变。
