# EXP004：多退化鲁棒自监督训练（正结果 KEEP）

- **Research Question**：多退化训练能否修复 EXP003 发现的退化失配（Gaussian 模型在 Box/Fermi 上损害重建）？
- **Hypothesis**：在自监督训练时从 {Gaussian, Box, Fermi} 随机采样退化核，让网络学习恢复多种退化，提升失配鲁棒性。
- **Method**：扩展 `SMOREDataset` 支持多退化（filter_name 可为列表），训练 IXI002_multideg（3000 步，其余同基线）。
- **Results（IXI002，r=2，ΔPSNR vs 各自 Sinc）**：

| 实际退化 | Sinc | Gaussian 模型（EXP003） | 多退化模型（EXP004） |
|---|---:|---:|---:|
| Gaussian | 28.631 | 28.904 (+0.27) | **29.043 (+0.41)** |
| Box | 29.993 | 28.360 (**-1.63**) | **29.992 (≈0.00)** |
| Fermi | 29.817 | 28.573 (**-1.24**) | **29.913 (+0.10)** |

- **Conclusion**：多退化训练**修复了失配**——Box 从 -1.63→≈0、Fermi 从 -1.24→+0.10，
  且 Gaussian 条件反而略升（+0.27→+0.41）。
- **Decision**：**KEEP**。多退化自监督训练是有效且低成本的鲁棒性改进。
- **局限**：单开发受试者（IXI002）；需在更多受试者验证。
- **代码**：`src/self_supervised_dataset.py`（多退化）、`scripts/train_smore.py`（--filters）、
  `improvements/scripts/degradation_robustness_pilot.py`。
