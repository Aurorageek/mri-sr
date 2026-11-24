# 文献阅读笔记

> 规范：记录 Title / Authors / Year / Venue / DOI-URL / Core Idea /
> Relevant Experiments / Limitations / Relationship to Our Research。
> 只读摘要的必须明确标注。禁止编造文献。

## 1. SMORE（核心参考）

- **Title**：SMORE: A Self-Supervised Anti-Aliasing and Super-Resolution Algorithm for MRI Using Deep Learning
- **Authors**：Zhao et al.
- **Year / Venue**：2021 / IEEE Transactions on Medical Imaging
- **DOI**：10.1109/TMI.2021.3056532（正式开展写作前需再次核实）
- **Core Idea**：利用 MRI in-plane 高分辨率平面作自监督目标，训练 2D 网络学一维
  去模糊，再迁移到 through-plane 方向；含多方向重建与 SAA 去混叠。
- **Our Relationship**：本项目为简化版（单方向、无 SAA/FBA）。Day 1–6 已验证
  SMORE > Sinc 的稳健性。

（其余文献在阅读后追加；当前阶段优先做退化对齐的数值分析，暂不扩大文献检索。）
