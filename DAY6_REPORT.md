# Day 6：MRI 超分辨率科研实验审查与阶段性总结

> 生成日期：2026-11-24。本报告基于项目真实代码、配置、日志、曲线与重建结果，
> 不引入新模型、不新增损失、不盲目重训。所有结论均来自真实证据。

---

## 1. 项目研究目标

复现 **SMORE(3D)**（Zhao et al., IEEE TMI 2021）的自监督 MRI through-plane
超分辨率：从高分辨率三维 MRI 模拟层间低分辨率采集，利用 MRI 自身 in-plane 高分辨率
平面构造自监督训练样本，用简化 EDSR 恢复 through-plane 分辨率，并评估 Sobel
Gradient Loss 对结构保持的影响。

## 2. 当前代码架构

```
src/
├── dataset.py               # NIfTI 读取/检查/归一化/三视图
├── degradation.py           # through-plane k-space 退化 + Sinc 插值 + 1D 低通
├── edsr.py                  # 简化 EDSR（8块/64通道/无BN/无上采样/全局残差）
├── self_supervised_dataset.py # SMORE 自监督样本（in-plane 切片模糊→清晰）
├── losses.py                # SobelGradientLoss + L1+λ·Grad 组合
└── metrics.py               # PSNR/SSIM/MAE/Gradient-MAE/brain_mask
scripts/
├── fetch_ixi_partial.py     # 从官方 tar 流式获取 IXI T1
├── inspect_mri.py / check_env.py
├── eval_interpolation.py    # Day 2 退化 + Sinc baseline
├── train_smore.py           # Day 3/4/5 训练（TensorBoard/CSV/PNG 记录）
├── infer_smore.py           # 三维推理
├── eval_smore.py / eval_multisubject.py / analyze_structure.py / plot_training_curves.py
configs/                     # degradation.json / smore_baseline.yaml / day5_experiment.yaml
results/                     # checkpoints / tensorboard / logs / metrics / figures
```

## 3. Day 1–5 已完成的实际工作

| Day | 内容 | 状态 |
|---|---|---|
| 1 | 10 例 IXI T1 获取、NIfTI 读取/检查、三视图 | 完成 |
| 2 | r=2 through-plane（R 轴）k-space 退化 + Sinc baseline + PSNR/SSIM | 完成 |
| 3 | 简化 SMORE(3D) 自监督训练 + 三维推理 + 与 Sinc 对比 | 完成 |
| 4 | Sobel Gradient Loss、训练曲线记录（TensorBoard/CSV/PNG）、单受试者初步消融 | 完成 |
| 5 | 多受试者（8 测试受试者）正式对比 + 消融 + 结构保真分析 | 完成 |
| 6 | 科研审查、正确性修正、阶段性总结（本报告） | 完成 |

## 4. 实验协议

- **数据**：10 例 IXI T1；开发集 = IXI002/IXI012，独立测试集 = 8 例
  （IXI013/014/015/016/017/019/020/021）。见 `configs/day5_experiment.yaml`。
- **退化**：r=2，through-plane 轴 = **R（canonical axis 0，1.2mm 厚层方向）**，
  gaussian 抗混叠 + 下采样（Day 2 一致）。
- **方法**：Sinc、SMORE-L1（λ=0）、SMORE-Grad（λ=0.10）。
- **训练**：EDSR 8×64，Adam lr=1e-4，batch 16，3000 步，seed 42，FP32。
- **评价**：统一 brain mask（HR > 0.01·P99.9，腐蚀×2），PSNR data_range = HR 前景
  动态范围，指标在原始强度域、mask 内计算；HR 只在最终评价读取。

## 5. 主实验结果（8 名独立测试受试者，mean ± std）

| Method | PSNR (dB) | SSIM | MAE | GradMAE（边缘误差） |
|---|---:|---:|---:|---:|
| Sinc | 30.50 ± 2.19 | 0.9291 | 68.80 | 489.96 |
| SMORE-L1 | 30.81 ± 2.28 | 0.9338 | 64.18 | 375.77 |
| SMORE-Grad (λ=0.10) | 30.82 ± 2.28 | 0.9340 | 64.11 | 374.94 |

**配对差异（Wilcoxon signed-rank，n=8）：**

- SMORE-L1 vs Sinc：ΔPSNR **+0.316 dB**（p=0.008）、ΔSSIM **+0.0047**（p=0.008）、
  ΔMAE **-4.63**（p=0.008）、ΔGradMAE **-114.2**（p=0.008，边缘误差 **-23%**）。
  **8/8 受试者全部改善，稳健显著。**
- SMORE-Grad vs SMORE-L1：ΔPSNR +0.007 dB（p=0.016）、ΔMAE -0.07（p=0.008）、
  ΔGradMAE -0.83（p=0.039）。**统计可检出但幅度极小、无实际意义。**

逐例指标见 `results/metrics/per_subject.csv`；汇总见 `summary.csv`；配对差异见
`ablation.csv`。

## 6. 消融结果（λ 扫描，开发集）

| λ | IXI002 自监督 val L1 | IXI012 自监督 val L1 |
|---|---:|---:|
| 0 | 0.001286 | 0.001492 |
| 0.01 | 0.00130 | 0.001655 |
| 0.05 | 0.00131 | 0.001744 |
| 0.10 | 0.00134 | 0.001334 |

λ 选择**弱/模糊**：无一致最优，取最强约束 λ=0.10 作代表。测试集上 Grad 与 L1 差异
（+0.007 dB）印证了梯度损失相对 L1 冗余的结论。

## 7. 训练曲线（真实日志）

- TensorBoard：`tensorboard --logdir .\results\tensorboard --port 6006`
- PNG：`results/figures/training_curves/fig1_train_total.png`、
  `fig2_val_total.png`、`fig3_l1.png`、`fig4_gradient.png`
- CSV：`results/logs/csv/<exp>_train.csv`、`_val.csv`

**收敛/过拟合**：train 与 val L1 均从 ~0.005 降至 ~0.0015 并平台化；未见明显
过拟合间隙；不同 λ 的 L1/Gradient 曲线高度重叠。注意 train 是每 10 步瞬时 batch
loss（较噪），val 是全集均值，故 train 偶高于 val 属正常。

## 8. 失败案例分析

- **SMORE 相对 Sinc 提升最大**：IXI014（ΔPSNR +0.517）、IXI013（+0.469）。
- **SMORE 表现最差**（绝对 PSNR）：IXI016（28.63）、IXI021（28.74）、IXI020（29.13）
  ——均为 Guys 序列、HR 质量较低的受试者；但 SMORE 仍相对 Sinc 有正增益。
- **Gradient Loss 改进最明显**：IXI021（ΔPSNR +0.013）、IXI013（+0.012）。
- **Gradient Loss 性能下降案例**：IXI017（ΔPSNR -0.0002、ΔGradMAE +0.70，轻微回退）、
  IXI015（SSIM -0.00005）。均属噪声级差异。

结构保真图（统一窗口、含局部放大、绝对误差与梯度误差）见
`results/figures/error_maps/`。观察：SMORE 明显比 Sinc 锐利、边缘误差下降，
但 Grad 与 L1 的误差图几乎无法区分，未见过度锐化或不真实高频纹理。

## 9. 已发现的科研问题（本次审查）

### 9.1 已确认并修复的正确性问题

1. **MAE 曾被误算为 RMSE（严重）**
   - 位置：`src/metrics.py::evaluate_with_edge` 中 `mae = mse ** 0.5`。
   - `mse ** 0.5` 是 RMSE，不是 MAE（`mean |ref-test|`）。
   - 影响：Day 4 与 Day 5 的 "MAE" 列实际是 RMSE（被系统性高估）。**PSNR/SSIM/GradMAE
     不受影响。**
   - 已修复：新增 `mae()`，`evaluate_with_edge` 改用它；Day 5 指标已用真实 MAE 重算。
   - 修正后 Sinc MAE 68.80、SMORE-L1 64.18、SMORE-Grad 64.11（原 RMSE 为 104.0/100.3/100.2）。

2. **Axial/Coronal/Sagittal 切面命名错误（中等）**
   - canonical RAS+：axis 0=R、1=A、2=S。fix S(axis2) 得到 **R-A 平面 = Axial**，
     但 Day 2/3/4/5 的可视化脚本把 `fix S` 标成了 "sagittal"（矢状面）。
   - 正确映射：fix R→sagittal(A-S)、fix A→coronal(R-S)、fix S→axial(R-A)。
   - 影响：`eval_interpolation.py`、`eval_smore.py`、`analyze_structure.py` 生成的
     所谓 "sagittal" 图实际是 axial（轴状面）。**只影响可视化标签，不影响任何 3D 指标。**
   - 已修复代码并重生成 Day 5 结构保真图（`_axial_*.png`）；Day 2/4 的旧图仍存在
     但标签有误（已在本报告记录，不再重跑）。

### 9.2 已确认但属设计说明（非 bug）

3. **Gradient MAE 为各向同性 3D 梯度幅值误差**：`scipy.ndimage.sobel`（无 axis）
   计算三个方向的梯度幅值合成，包含 through-plane（R）方向但不单独加权。这是合理的
   "结构保真代理"，但**未专门隔离被退化方向**。如需更针对性的分析，应单独报告
   through-plane 方向的梯度误差。

4. **退化模型是幅度图模拟**：Day 2 对幅度图做 FFT 高斯低通 + 下采样，等价于理想
   带限 + 高斯模糊的简化，**不等同于真实复数 k-space 少采样**。README 已声明。

### 9.3 已确认正确的项

- 退化轴 = R（与 Day 1 数据方向结论一致）。✓
- 自监督训练**无 HR 泄漏**：训练只读 `_sinc_r2.nii.gz`（g），HR 只在最终评价读取；
  有源码级测试 `test_smore.py::test_no_hr_in_training`。✓
- 不同方法共用同一 8 名测试受试者、同一 LR 输入（sinc_r2）、同一 mask、同一
  data_range 与强度域。✓
- 超参数（λ）在独立开发集（IXI002/IXI012）上、用自监督 val L1 选择，未用 HR 或
  测试集。✓

## 10. 当前实现与 SMORE 原论文的区别

1. 只做**单方向**推理（R-A 平面，fix S）；原论文含多方向旋转集成。
2. 无 **SAA 去混叠网络**、无 Fourier Burst Accumulation、无 Physics Consistency Loss。
3. 网络更小（8 残差块 / 64 通道），非论文完整超参数。
4. 退化轴 = R（本数据矢状位存储），论文默认 through-plane = Z。
5. 退化模型为幅度图高斯 + 下采样简化，非复数 k-space。

## 11. 下一阶段研究建议（基于证据）

回答三个关键问题：

**Q1：Gradient Loss 是否值得继续研究？——不建议继续 Sobel 梯度损失。**
证据：8 受试者上相对 L1 仅 +0.007 dB、边缘误差 -0.2%，与 L1 高度冗余（Sobel 是线性
算子，L1+λ·Sobel-L1 等价于对像素差做边缘加权）。若要研究"结构保持"，应改用
**非线性/非冗余**约束（感知损失 VGG、SSIM/结构损失），才可能产生可分离效果。

**Q2：当前 SMORE Baseline 是否可靠？——可靠。**
证据：8/8 受试者一致优于 Sinc，PSNR +0.32 dB、边缘误差 -23%，配对检验 p=0.008。
可作为课程论文的核心结论。

**Q3：退化模型失配是否值得进一步实验？——值得，但优先级次之。**
当前是幅度图简化模拟；真实采集含复数 k-space、层选轮廓、噪声。若时间允许，可做
退化鲁棒性实验（不同噪声水平/不同模糊核），验证方法泛化性；但这不改变"SMORE>Sinc"
与"Gradient Loss 冗余"的现有结论。

**综合建议**：D（冻结 SMORE vs Sinc 结论并开始英文论文整理）+ 若有余力做 B
（换用非冗余结构损失）或 A（扩大样本到 20+ 例以获得更稳的显著性）。

---

## 附：本次修改文件清单

- 修改：`src/metrics.py`（新增 `mae()`，修复 MAE/RMSE）、
  `scripts/eval_smore.py`、`scripts/analyze_structure.py`、
  `scripts/eval_interpolation.py`（切面命名 sagittal→axial）。
- 重算：`results/metrics/per_subject.csv`、`summary.csv`、`ablation.csv`、
  `day5_metrics.json`（MAE 修正）。
- 重生成：`results/figures/error_maps/*_axial_*.png`（移除误标的 `*_sagittal_*.png`）。
- 新增：本报告 `DAY6_REPORT.md`。
