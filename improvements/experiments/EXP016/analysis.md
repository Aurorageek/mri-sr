# EXP016：原论文对齐的退化模型与 Baseline A

## 与原始 SMORE 的差异核查

| 项目 | 原论文 | 本项目此前 | 本轮处理 |
|---|---|---|---|
| 退化模型 | `rect(r·u) × Fermi(u;r)`，Fermi 宽度 10/L | Gaussian/Box/Fermi 单一核 | **新增 `paper` 滤波器，按式 (1) 实现** |
| 网络 | EDSR（论文未给完整超参） | 简化 EDSR 8 块/64 通道 | 保留简化 EDSR，明确声明非完整复现 |
| Patch | 小 patch（SAA 用 32×32） | 32×32 | 一致 |
| 旋转增强 | 7 个旋转角（θ=nπ/12），留一验证 | 仅翻转 | **未复现**（差异已声明） |
| 推理方向 | 单方向（M=1）为主 | R-A 单方向 | 一致 |
| 多方向 + FBA | M=2/7 + FBA | 未实现 | 本轮实现 M=2 FBA 对照 |
| 优化器 | Adam, lr=1e-4 | Adam 1e-4 | 一致 |

## 实现

`src/degradation.py` 新增 `"paper"` 滤波器：
`mask = rect(|f|<=1/(2r)) × 1/(1+exp((|f|-1/(2r))/(10/L)))`，L=该轴长度。
单元测试：`improvements/tests/test_paper_degradation.py`（4/4 通过）。

## Baseline A

- 训练退化 = `paper`；推理 = R-A 单方向（fix S）。
- 在 IXI002/IXI012/IXI013/IXI022 上各训练 3000 步（seed 42，与历史一致）。
- checkpoint：`results/checkpoints/<subject>_paperaligned/best.pt`。
- 具体 PSNR/SSIM 见 EXP018 汇总表（Single-A 行）。

## 尚未复现的模块

1. 7 旋转训练增强与留一验证。
2. 原论文完整 EDSR 超参数。
3. SAA（仅 2D 协议需要）。
4. 原论文数据集与临床评价。
