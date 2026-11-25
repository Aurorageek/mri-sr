# improvements/：自主科研改进研究目录

本目录用于 Day 6 之后的研究循环，与 Day 1–6 的 Baseline（`src/`、`scripts/`、
`results/`）隔离。原始数据、EDSR、MRI 读取、评价等模块直接复用。

## 查看 TensorBoard

```powershell
tensorboard --logdir .\results\tensorboard --port 6006
```

改进实验若使用独立日志，同样可通过 TensorBoard 查看（见具体实验目录）。

## 目录结构

```
improvements/
├── README.md              # 本文件
├── RESEARCH_PLAN.md       # 研究计划与方向优先级
├── RESEARCH_LOG.md        # 每轮实验的研究日志（含负结果）
├── EXPERIMENT_PROTOCOL.md # 冻结的基线、评价协议、数据隔离
├── CURRENT_STATUS.md      # 当前状态与下一轮计划
├── BEST_METHOD.md         # 当前最佳方法
├── methods/               # 仅在真正开展某方向时创建对应子目录
├── configs/ scripts/ tests/
├── experiments/EXPxxx/    # 每轮实验（config/hypothesis/metrics/analysis）
├── results/               # experiment_registry.csv / comparison / ablation / summary
├── figures/               # training_curves / comparisons / error_maps / robustness
├── references/            # literature_notes.md
└── reports/               # 阶段性研究报告
```

## 研究循环

Observation → Hypothesis → Experimental Design → Implementation → Pilot →
Evaluation → Decision（KEEP / DISCARD / INCONCLUSIVE / FAILED）→ 记录并更新 Git。

## Phase 4 复现命令（原论文对齐 + 跨平面可靠性）

```powershell
# 1) 生成原论文对齐 LR（H = rect(r·u) × Fermi(u;r)，Eq.1）
python improvements/scripts/generate_lr.py --subject IXI002-Guys-0828-T1 --filter paper
# 2) 训练 paper-aligned Baseline A
python scripts/train_smore.py --loss l1 ^
  --g-path data/processed/IXI002-Guys-0828-T1/IXI002-Guys-0828-T1_sinc_r2_paper.nii.gz ^
  --filters paper --experiment-name IXI002-Guys-0828-T1_paperaligned
# 3) 跨平面分歧与误差关系（EXP017）
python improvements/scripts/cross_plane_analysis.py
# 4) RG-SMORE 与对照（固定缩放/FBA/梯度门控，EXP018）
python improvements/scripts/reliability_guided.py
# 5) paper 退化单元测试
python improvements/tests/test_paper_degradation.py
```

说明：`src/degradation.py` 新增 `"paper"` 滤波器；RG/跨平面分析只做推理与后处理，
不新增训练。
