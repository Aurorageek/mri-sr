# 文献阅读笔记

> 规范：记录 Title / Authors / Year / Venue / DOI-URL / Core Idea /
> Relevant Experiments / Limitations / Relationship to Our Research。
> 只读摘要或检索片段的必须明确标注。禁止编造文献。

## 1. SMORE（核心参考，已读 PMC 开放全文 Methods 片段）

- **Title**：SMORE: A Self-Supervised Anti-Aliasing and Super-Resolution Algorithm for MRI Using Deep Learning
- **Authors**：Can Zhao, Blake E. Dewey, Dzung L. Pham, Peter A. Calabresi, Daniel S. Reich, Jerry L. Prince
- **Year / Venue**：2021 / IEEE Transactions on Medical Imaging, 40(3):805–817
- **DOI / URL**：10.1109/TMI.2020.3037187；PMC8053388（开放全文）
- **Core Idea**：单受试者自监督（SSR）。用 in-plane 高分辨率平面构造（模糊→清晰）训练对，
  学到的一维去模糊迁移到 through-plane；EDSR 作为回归网络。
- **退化模型（原文 Eq. (1)，已核实原文）**：
  `H(u;r) = rect(r·u) × Fermi(u;r)`，其中
  `Fermi(u;r) = 1 / (1 + exp((|u| − 1/(2r)) / (10/L)))`，
  L 为该方向图像尺寸；r 为 through-plane 与 in-plane 分辨率之比。
- **多方向与 FBA（已核实）**：早期工作用 M=2（coronal + sagittal）；论文比较 M=1/2/7，
  M=7 时旋转角 θ = nπ/12, n=0..6；其余实验 M=1。多个估计用 Fourier burst
  accumulation（FBA）融合（SMORE 引用其文献 [6]；FBA 的标准形式为频域按
  |F_i|^p / Σ|F_j|^p 加权后求和）。
- **训练细节（已核实片段）**：Adam，lr=1e-4；旋转增强；用一个旋转角的样本作验证；
  小 patch；SAA 网络用 32×32 patch。
- **Relationship**：本项目以其为 Baseline。Phase 4 实现了 `rect×Fermi` 的 paper 对齐
  退化，并研究其未解决的"可靠性/错误高频"问题。

## 2. iSMORE（检索片段，未读全文）

- **Title**（检索标题）：iSMORE / iterative self-supervised SR（Springer 章节，
  检索结果链接 10.1007/978-3-030-32778-1_14）
- **Core Idea（据检索摘要）**：在 SMORE 基础上加入 **Sobel 边缘损失**与迭代策略。
- **Relationship / 重要定位**：我们 Day 4 的 Sobel Gradient Loss 与 iSMORE 的 edge loss
  高度重叠，**不能作为原创贡献**。该方向已在 Day 4/5 判定为与 L1 冗余并 DISCARD。
- **注意**：仅阅读检索摘要与标题，未读全文，引用前需核实。

## 3. Pushing the Limits of Zero-Shot Self-Supervised SR of Anisotropic MR Images（检索片段）

- **URL**：PMC11613508
- **Core Idea（据检索摘要）**：以 SMORE 为基线，探索 wavelet/FFT 损失、对抗损失、
  LR re-gridding（B-spline、spectral padding）等改进。
- **Relationship**：说明"简单换损失/换重采样"已有大量工作；本项目应避免重复，
  转向跨平面可靠性与错误结构分析。

## 4. Fourier Burst Accumulation（FBA）

- **来源**：SMORE 引用 [6]；标准定义为 Delbracio & Sapiro 提出的多帧频域加权融合。
- **Formula used in this project**：`F = Σ_i w_i F_i`，`w_i = |F_i|^p / Σ_j |F_j|^p`，
  p=3；再 IFFT。**注意**：这是标准 FBA 形式，非本项目原创。
- **Relationship**：作为 Phase 4 的对照方法之一（M=2 的两个正交平面估计）。

## 5. 待核实 / 未阅读

- 关于 "prediction disagreement as uncertainty" 的医学图像 SR 工作：本项目在 Phase 4
  只做了内部模拟实验验证，尚未系统检索该方向是否已有同名方法；正式论文写作前
  必须补充检索，避免把已有方法包装成原创。
