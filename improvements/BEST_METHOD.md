# 当前最佳方法

> 截至 EXP004：**多退化自监督训练（SMORE-L1 + Multi-Degradation）** 是当前最有
> 价值的方法（提升鲁棒性，待多受试者验证）。

## 方法

在 SMORE 自监督训练时，从 {Gaussian, Box, Fermi} 随机采样退化核构造训练输入，
让网络学习恢复多种 MRI 退化，而非固定单一退化。

## 与 Baseline 的区别

Baseline（SMORE-L1）用固定 Gaussian 退化；改进版在训练时随机采样多种退化，
推理时不改变网络结构、不改变计算开销。

## 当前证据（EXP004，IXI002 单开发受试者）

- 修复退化失配：Box 条件从 -1.63 dB（低于 Sinc）→ ≈0；Fermi 从 -1.24 → +0.10。
- Gaussian 条件反而略升：+0.27 → +0.41 dB。
- 训练/推理开销与 Baseline 相同（无新增参数量）。

## 复现命令

```powershell
python scripts/train_smore.py --loss l1 --g-path data/processed/<s>/<s>_sinc_r2.nii.gz \
    --filters gaussian box fermi --experiment-name <s>_multideg
```

## 已知局限

- 仅 IXI002 单受试者验证，需扩展。
- 未测试 r=4、更多退化类型（噪声、不同截止频率）。

## 历史最佳（被替代）

- SMORE-L1（固定 Gaussian）：PSNR 30.81 vs Sinc 30.50，8 受试者，但失配条件下
  会损害重建（EXP003 发现）。
