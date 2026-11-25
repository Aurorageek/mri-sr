# 当前最佳方法

> 更新（Phase 4）：**Paper-Aligned SMORE + M=2 平均/FBA（+ 可选噪声增强）**。

## 组成

1. **退化模型**：SMORE Eq.(1) `H(u;r)=rect(r·u)·Fermi(u;r)`（`filter="paper"`）。
   paper 对齐退化下，单方向 SMORE 相对 Sinc 提升 **+0.78~+1.02 dB**（4 受试者）。
2. **多方向融合**：M=2（R-A 与 R-S 推理）取平均或 FBA；相对单方向仅 +0.003~+0.016 dB
   （两方向高度相关），但稳定不为负。
3. **噪声鲁棒（可选）**：训练输入加 Rician 噪声 σ=0.02（近零干净代价）或 0.05
   （高噪声更强）。干净训练的 SMORE 在噪声下会低于 Sinc。

## 明确不采用

- **RG-SMORE（可靠性门控）**：EXP018 负结果，劣于平均与能量匹配固定缩放。
- **Sobel Gradient Loss**：与 L1 冗余（且与 iSMORE 重叠，非原创）。
- **Uniform-Multi / 自适应退化采样**：Phase 2/3 已证伪。

## 证据

- paper-aligned 单方向 vs Sinc：+0.78/+0.91/+1.02/+0.89 dB（IXI002/012/013/022）。
- M=2 Average/FBA：29.675/34.179/35.047/30.526 dB（单方向 29.659/34.171/35.040/30.521）。
- 噪声：见 `experiments/EXP019/analysis.md`。

## 复现

```powershell
python improvements/scripts/generate_lr.py --subject IXI002-Guys-0828-T1 --filter paper
python scripts/train_smore.py --loss l1 --g-path data/processed/IXI002-Guys-0828-T1/IXI002-Guys-0828-T1_sinc_r2_paper.nii.gz --filters paper --experiment-name IXI002_paperaligned
python improvements/scripts/reliability_guided.py   # 含 Average/FBA 对照
```

## 局限

- 仅 r=2；独立验证受试者有限。
- 旋转增强（原论文 7 角度）未复现。
- 噪声方法仅 2 名受试者 pilot。

## 历史方法（已被替代）

- SMORE-Fixed-Box（Phase 3，Gaussian 模拟协议下）：mean +0.244 dB，12/12 为正。
  在 paper 对齐协议下未重测，不再作为首选。
