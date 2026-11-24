# 当前最佳方法

> 最终方法（Phase 3 冻结）：**SMORE-Fixed-Box**。

## 方法

自监督训练时用 **Box（理想矩形截断）低通核**构造 in-plane 训练退化（而非默认的
Gaussian），网络结构、训练预算、推理方式均不变。

## 与 SMORE Baseline 的区别

基线用 Gaussian 模糊核构造自监督训练对；本方法改用 Box 核。
无额外参数、无额外推理开销、无需退化估计或多退化采样。

## 证据

**开发集（4 受试者 × 3 条件）**：平均 ΔPSNR **+0.244 dB**，最差 **+0.140 dB**，12/12 为正。

**独立测试（4 名全新受试者 × 3 条件）**：

| 方法 | mean ΔPSNR | worst | 优于 Sinc 比例 |
|---|---:|---:|---:|
| Fixed-Gaussian | -0.904 | -1.811 | 4/12 |
| **Fixed-Box** | **+0.244** | **+0.062** | **12/12** |

- 匹配条件（Gaussian）：Fixed-G +0.26 略优于 Fixed-Box +0.17。
- 失配条件（Box/Fermi）：Fixed-Box 全部为正，Fixed-G 全部大幅为负。

## 复现命令

```powershell
python scripts/train_smore.py --loss l1 \
    --g-path data/processed/<subject>/<subject>_sinc_r2.nii.gz \
    --filters box --experiment-name <subject>_indep_fixedbox
python improvements/scripts/eval_independent_robustness.py
```

## 已知局限

- 仅 r=2 有明确收益；r=4 增益消失。
- 机制（Box 更易迁移）为假设，未理论证明。
- 独立测试仅 4 名受试者。

## 被替代的历史方案

- SMORE-Uniform-Multi（Phase 2）：mean +0.171（开发集），差于 Fixed-Box。
- SMORE-Fixed-Gaussian：仅在匹配条件有效，失配时严重下降。
