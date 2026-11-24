# Phase 3 研究报告：方法优化与正式实验验证

## 1. 前两轮关键实验结论的复核

- **EXP011 协议审计**：逐 checkpoint 读取 `config.data.g_path`，确认 EXP007 全部是
  标准 Protocol B（每个模型都用其对应实际退化的 g 自身训练），未复用历史结果。
- 前轮"多退化修复失配"结论方向正确，但**比较对象不完整**（只与 Fixed-Gaussian 比），
  本轮补齐了 Fixed-Box / Fixed-Fermi 基线。

## 2. 完整基线比较（EXP012，IXI002，r=2，ΔPSNR vs Sinc）

| actual | Sinc | Fixed-G | Fixed-Box | Fixed-Fermi | Uniform-Multi |
|---|---:|---:|---:|---:|---:|
| Gaussian | 28.631 | +0.273 | +0.222 | +0.224 | +0.412 |
| Box | 29.993 | **-1.637** | +0.340 | +0.192 | +0.001 |
| Fermi | 29.817 | **-1.236** | +0.291 | +0.248 | +0.100 |
| **mean** | — | -0.867 | **+0.284** | +0.221 | +0.171 |
| **worst** | — | -1.637 | **+0.222** | +0.192 | +0.001 |

**关键修正**：退化失配是 **Gaussian 特有**；Fixed-Box/Fermi 天然鲁棒；
**Uniform-Multi 并不优于 Fixed-Box**，甚至更差。

## 3. 退化估计的可行性分析（EXP014）

Gaussian/Box/Fermi 三种 LR 的沿 R 轴归一化功率谱相关系数 **> 0.99999**，几乎不可区分
（截止频率相同，仅过渡形状不同）。→ 基于 LR 频谱的退化估计**不可行**，自适应采样
缺乏可靠依据，应放弃。

## 4. 自适应退化方法及实际结果

未实现（EXP014 已证明其前提不成立）。这是有依据的负结果，避免了无意义开发。

## 5. 残差与高频结构分析（EXP009 复核）

- Fixed-G 的残差在失配条件**单调有害**（α↑→PSNR↓），并在 Box 条件产生更高
  Through-plane 梯度误差（引入伪边缘）。
- Fixed-Box 的残差中性偏正，未产生明显伪高频。
- 因此"失配危害"来自 Gaussian 的过度锐化，而非多退化本身的价值。

## 6. 方法对比与消融

| 消融 | 对比 | 结果 |
|---|---|---|
| 多退化是否必要 | Fixed vs Uniform-Multi | Multi 反而差于 Fixed-Box |
| 单退化是否足够 | Fixed-Box vs Multi | Fixed-Box 更优（+0.284 vs +0.171）|
| 自适应是否必要 | 频谱分析 | 退化不可区分，无依据 |
| 失配来源 | Gaussian vs Box/Fermi 核 | 仅 Gaussian 失配 |

## 7. 最终方法选择依据

**SMORE-Fixed-Box**（自监督训练用 Box 核构造退化）：
- 开发集 4 受试者：平均 **+0.244 dB**，最差 **+0.140 dB**，12/12 为正。
- 独立测试 4 受试者：平均 **+0.244 dB**，最差 **+0.062 dB**，**12/12 为正**。
- 无额外参数、无额外推理开销，比多退化更简单且更稳健。

## 8. 独立受试者实验结果（EXP015，新受试者 IXI022–025）

| subject | Sinc | G:gauss | G:box | G:fermi | B:gauss | B:box | B:fermi |
|---|---:|---:|---:|---:|---:|---:|---:|
| IXI022 | 29.604 | +0.243 | -1.611 | -1.194 | +0.062 | +0.167 | +0.286 |
| IXI023 | 28.745 | +0.240 | -1.678 | -1.313 | +0.220 | +0.258 | +0.338 |
| IXI024 | 28.147 | +0.238 | -1.663 | -1.251 | +0.221 | +0.370 | +0.290 |
| IXI025 | 28.919 | +0.321 | -1.811 | -1.365 | +0.180 | +0.291 | +0.238 |

汇总（n=12）：Fixed-G mean **-0.904**（worst -1.811，4/12 为正）；Fixed-Box mean
**+0.244**（worst +0.062，**12/12 为正**）。

## 9. 已完成的论文级图表

- `improvements/figures/paper/fig3_mismatch_heatmap.png`（3×2 ΔPSNR 热力图）
- `improvements/figures/paper/fig4_robustness_independent.png`（独立受试者稳健性）
- `improvements/figures/paper/fig5_reconstruction_IXI022-..._{coronal,axial}.png`
  （HR/Sinc/Fixed-G/Fixed-Box + 局部放大 + 误差图）

## 10. 代表性失败案例

- Fixed-Gaussian 在 Box 条件：PSNR 比 Sinc 低约 1.6 dB，误差图与梯度误差显示
  过度锐化/振铃（见 EXP009/EXP012）。
- r=4：SMORE 增益几乎消失（Gaussian 上 +0.005 dB），属方法适用范围边缘。

## 11. 研究局限

1. 独立测试仅 4 名受试者（paper 建议扩到 8–10）。
2. 仅 r=2 有明确收益；r=4 收益小。
3. 模拟退化为幅度图简化，非复数 k-space。
4. Fixed-Box 的"为何更鲁棒"仍属机制假设（Box 硬截断的锐化更易迁移），未做理论证明。

## 12. 可复现运行命令

```powershell
# 生成某退化条件的 LR
python improvements/scripts/generate_lr.py --subject IXI022-Guys-0701-T1 --filter box
# 训练 Fixed-Box（Protocol B）
python scripts/train_smore.py --loss l1 --g-path data/processed/IXI022-Guys-0701-T1/IXI022-Guys-0701-T1_sinc_r2.nii.gz --filters box --experiment-name IXI022-Guys-0701-T1_indep_fixedbox
# 评估
python improvements/scripts/eval_independent_robustness.py
python improvements/scripts/make_paper_figures.py
```

## 13. 最新 Git commit 与 GitHub 地址

- 分支：`research/mri-sr-improvements`
- 地址：https://github.com/Aurorageek/mri-sr/tree/research/mri-sr-improvements
- 最新 commit 见提交记录（本报告随 EXP011–015 一同提交）。

## 14. 距正式提交尚缺的工作

1. 将独立测试扩到 **8–10 名**（当前 4 名）以获得统计功效。
2. 补充 SSIM / Through-plane Gradient / 3D Gradient 的完整逐受试者表与置信区间。
3. 完成英文论文正文（PAPER_EXPERIMENTS.md 已提供实验素材）。
4. 可选：多随机种子（seed 稳定性）与 r=4 的补充消融。
