# MRI-SR：SMORE 自监督 MRI 超分辨率（Day 1–4）

本项目为医学大数据与人工智能课程项目，研究方向是 MRI 超分辨率重建，
参考 **SMORE: A Self-Supervised Anti-Aliasing and Super-Resolution Algorithm
for MRI Using Deep Learning**（IEEE TMI, 2021）的 SMORE(3D) 思路。

**当前进度**：

- Day 1：数据获取、NIfTI 读取/检查/归一化、三视图（已完成）。
- Day 2：through-plane 各向异性退化 + Sinc 插值 baseline + PSNR/SSIM（已完成）。
- Day 3：简化 SMORE(3D) 自监督训练 + 推理 + 与 Sinc 对比（已完成）。
- Day 4：Sobel Gradient Loss、训练曲线记录（TensorBoard/CSV/PNG）、初步消融（已完成）。
- 尚未开始：多方向集成、SAA 去混叠网络、扩大数据、正式消融。

---

## 1. 实际运行环境（2026-11-21 检查）

| 项目 | 实际值 |
|---|---|
| 操作系统 | Windows 11 Pro 10.0.22621 (AMD64) |
| Conda 环境 | `mri`（`C:\Users\Xuanyu Liu\miniconda3\envs\mri`） |
| Python | 3.11.18 |
| NumPy | 2.4.6 |
| NiBabel | 5.5.0（本次安装） |
| Matplotlib | 3.11.2（本次安装） |
| SciPy | 1.17.1（Day 2 安装） |
| scikit-image | 0.26.0（Day 2 安装，SSIM） |
| PyYAML | 6.0.4（Day 3 安装，配置） |
| PyTorch | 2.7.1+cu118（环境中已有，未改动） |
| CUDA (torch) | 11.8 |
| cuDNN | 9.1.0 |
| GPU | NVIDIA GeForce RTX 2080 Ti，11.0 GB，compute capability 7.5 |

GPU 冒烟测试（64×64 矩阵乘）通过。环境报告见 `results/env_report.json`。

> 未重新安装 Conda/Python/PyTorch，未升级已有环境；Day 1 安装了
> `nibabel`、`matplotlib`，Day 2 安装了 `scipy`、`scikit-image`。

---

## 2. 数据集来源

使用 **IXI Dataset**（官方页面：<https://brain-development.org/ixi-dataset/>）。

- 官方 T1 下载地址：
  <https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar>
- 该 tar 约 **4.84 GB**，服务器支持 HTTP Range。
- 本阶段**没有下载完整数据集**：`scripts/fetch_ixi_partial.py` 以流式方式
  读取 tar 开头，仅提取前 3 名受试者后主动断开连接（约 27 MB）。
- 原始 `.nii.gz` 文件保持原样，未修改、未重采样。
- 文件名、大小、SHA-256、shape、spacing 记录在
  `data/raw/ixi/manifest.json`，便于复现与完整性校验。
- IXI 数据仅用于学术研究，请遵守官方页面的使用条件并引用 IXI。

本阶段实际获取的 3 名受试者：

| 文件 | 大小 | Shape (原始存储) | Voxel spacing (mm) | Orientation |
|---|---:|---|---|---|
| `IXI002-Guys-0828-T1.nii.gz` | 6.62 MB | (256, 256, 150) | (0.9375, 0.9375, 1.200) | PSR |
| `IXI012-HH-1211-T1.nii.gz` | 9.65 MB | (256, 256, 150) | (0.9375, 0.9375, 1.200) | PSR |
| `IXI013-HH-1212-T1.nii.gz` | 9.43 MB | (256, 256, 150) | (0.9375, 0.9375, 1.200) | PSR |

> 未使用随机合成图像冒充 IXI，也未使用第三方镜像。
> 合成数组只用于 `tests/test_dataset.py` 的**函数单元测试**，与真实数据验证严格区分。

### 数据方向的重要发现

3 例数据均由官方 tar 直接解出，raw orientation 为 **PSR**：

- 原始 voxel axis 0 → Posterior，0.9375 mm
- 原始 voxel axis 1 → Superior，0.9375 mm
- 原始 voxel axis 2 → Right，1.2 mm（共 150 层）

即这批 IXI T1 是按**矢状位切片**存储的（150 层沿左-右方向，层厚 1.2 mm），
矢状面内分辨率 0.9375 mm；原始 affine 还带有约 7° 的斜切旋转。

`nib.as_closest_canonical()` 后的 RAS+ 视图：

- shape = **(150, 256, 256)**（轴置换，不是数据损坏）
- voxel spacing = **(1.2, 0.9375, 0.9375) mm**
- orientation = **RAS**

原始文件不受 canonical 操作影响（只作用于内存）。三视图均基于 canonical RAS+ 数据绘制，
解剖方向已通过实际查看图片确认。

**为什么这件事对 Day 2 很关键**：CANONICAL RAS+ 下 resolution 较低的轴是 **R（左右）**，
而不是通常 SR 论文里默认的 **S（上下，即 "Z"）**。Day 2 做各向异性退化时必须明确
选择沿哪个轴降采样，不能默认"所有 MRI 都是 1×1×1 mm"或"Z 就是层方向"。
详见第 8 节。

---

## 3. 目录结构

```text
MRI_SR/
├── data/
│   ├── raw/
│   │   └── ixi/                     # 3 例原始 IXI T1 + manifest.json（只读）
│   └── processed/                   # 后续退化模拟/重建结果（当前为空）
├── src/
│   ├── __init__.py
│   └── dataset.py                   # load_mri / inspect_mri / normalize_mri / orthogonal_slices
├── scripts/
│   ├── check_env.py                 # 环境与 GPU 检查
│   ├── fetch_ixi_partial.py         # 从官方 tar 流式提取少量受试者
│   └── inspect_mri.py               # 检查 + 三视图 PNG
├── tests/
│   └── test_dataset.py              # 合成数组单元测试（11 项）
├── results/
│   ├── env_report.json
│   ├── mri_inspection.json          # 3 例真实数据的结构化检查结果
│   └── figures/
│       ├── IXI002-Guys-0828-T1_orthogonal.png
│       ├── IXI012-HH-1211-T1_orthogonal.png
│       ├── IXI013-HH-1212-T1_orthogonal.png
│       └── diagnostic_raw_axis_slices.png
├── configs/                         # 预留（Day 2 起放退化/训练配置）
├── test.py                          # 用户原有文件，保持未改动
├── requirements.txt
└── README.md
```

所有路径均使用 `pathlib` / 相对项目根目录的默认值，兼容 Windows。
脚本无论从哪个工作目录调用，都会自行定位项目根目录。

---

## 4. 依赖安装

```powershell
conda activate mri
python -m pip install -r requirements.txt
```

`requirements.txt` 中的版本是本次实际验证过的版本。PyTorch 已存在，**不要重装**。

如果 shell 里没有 `conda` 命令，也可以直接使用环境解释器：

```powershell
& "C:\Users\Xuanyu Liu\miniconda3\envs\mri\python.exe" scripts\inspect_mri.py
```

---

## 5. 常用命令

```powershell
# 1) 环境、CUDA、GPU 检查（写入 results/env_report.json）
python scripts/check_env.py

# 2) 获取 3 例 IXI T1（已获取则自动跳过下载）
python scripts/fetch_ixi_partial.py --n 3

# 3) 检查 data/raw 下全部 MRI，生成 JSON + 三视图 PNG
python scripts/inspect_mri.py

# 4) 只检查指定文件、不生成图片
python scripts/inspect_mri.py data/raw/ixi/IXI002-Guys-0828-T1.nii.gz --no-figures

# 5) 合成数据单元测试
python tests/test_dataset.py
```

如果自动下载失败，可手动从官方页面下载 `IXI-T1.tar`，解压后把任意 2–3 个
`IXI*.nii.gz` 放到 `data/raw/ixi/`。不要使用来源不明的镜像数据。

---

## 6. 代码接口（`src/dataset.py`）

```python
load_mri(path, canonical=False) -> MRIVolume
    # data: 3D np.ndarray；affine / voxel_spacing / orientation / shape / header_info
    # canonical=True 时返回 RAS+ 视图，并记录 original_shape / original_orientation

inspect_mri(path) -> dict
    # shape、voxel spacing、orientation、affine、强度 min/max、P1/P50/P99、
    # NaN/Inf、is_valid_3d、canonical 前后对比，可直接 json.dump

normalize_mri(volume, method="percentile", ...) -> np.ndarray
    # 返回新的 float32 数组，不修改输入；支持 percentile / minmax / zscore

orthogonal_slices(volume, voxel_spacing, indices=None) -> dict
    # 输入必须是 canonical RAS+；返回 axial/coronal/sagittal 切片及 mm 单位 extent
```

### 归一化方法说明

默认 `method="percentile"`：

1. 只统计**前景**（`volume > 0`）体素的第 1 与第 99.5 百分位；
2. 线性拉伸到 `[0, 1]`，并裁剪到该范围；
3. 背景体素保持为 `0`。

这样做的原因：T1 MRI 背景为 0，强度分布高度偏斜，少数极亮体素（脂肪、血管、
伪影）会把脑组织的动态范围压得很窄。用前景分位数可以稳定对比度，并且每例独立
计算，不需要数据集级统计，适合自监督训练。若动态范围接近 0，函数会给出警告并
返回全零数组，避免除零；输入含 NaN/Inf 时直接抛出 `ValueError`。

注意：分位数裁剪会压掉最亮的 0.5% 体素，做 PSNR/SSIM 定量评价时应保持
GT 与预测使用**完全相同**的归一化设置，或在原始强度空间评价。

---

## 7. 本阶段已完成 / 未完成

### 已完成并实际验证

- [x] 项目目录结构（保留原有 `test.py`，未覆盖）
- [x] 使用 `mri` conda 环境，检查 Python / PyTorch / CUDA / GPU（含 GPU 实算冒烟测试）
- [x] 从 IXI 官方源获取 3 例真实 T1，记录来源、大小、SHA-256
- [x] NIfTI 读取与元数据检查（shape / spacing / orientation / affine / 强度 / NaN/Inf）
- [x] canonical RAS+ 前后方向与维度对比，原始文件不变
- [x] 3 例三视图 PNG 实际生成并人工查看确认解剖正常
- [x] `load_mri` / `inspect_mri` / `normalize_mri` / `orthogonal_slices` 接口
- [x] 合成数组单元测试 11/11 通过；真实数据检查 3/3 通过
- [x] README 与 requirements

### 尚未完成（后续阶段）

- [x] 各向异性退化模拟（through-plane k-space 低通 + 下采样）
- [x] Sinc 插值 baseline + PSNR/SSIM 定量评价
- [ ] SMORE(3D) 自监督样本构造
- [ ] EDSR 模型与训练
- [ ] Gradient Loss 与消融实验
- [ ] 学习型方法与 Sinc baseline 的对比
- [ ] 扩大受试者数量（当前仅 3 例，用于调试）

---

## 8. Day 2 前必须注意的数据问题

1. **退化轴的选择**
   - 数据原生厚层方向是 **R（左右）**，GT 层厚 1.2 mm；A/S 两轴为 0.9375 mm。
   - 若按常见论文写法直接在 canonical 的 **S (Z)** 轴降采样，得到的低分辨率
     方向与真实采集几何不一致（真实厚层在 R）。
   - 更贴近真实采集的做法：沿 **R** 轴把 1.2 mm 退化到 2.4 / 3.6 mm；
     但 GT 本身只有 1.2 mm，放大倍数受限（×2、×3）。
   - 也可以先重采样到 1 mm 各向同性再沿 S 退化（便于和论文对齐），但重采样
     会引入插值误差并改变"真实高分辨率"的含义，必须在报告中说明。
   - 建议 Day 2 先明确这条研究设计，再确定实验协议，不要默认 Z 就是层方向。

2. **体素间距 ≠ 真实采集分辨率**。0.9375 / 1.2 mm 是重建网格间隔，受层厚、
   层间距、部分容积效应、扫描协议影响。报告里不要把它写成"扫描仪分辨率"。

3. **原始数据是 PSR + 约 7° 斜切**。任何依赖轴顺序的操作都必须先 canonicalize，
   并记录轴置换；不要在 raw 数组上直接假设 `axis 2 = Z`。

4. **3 例数据都来自同一存储格式**，但不代表所有 IXI 受试者完全一致。扩大样本时
   仍应逐例运行 `inspect_mri.py` 检查 shape / spacing / orientation。

5. **定量评价的归一化一致性**：训练、验证、PSNR/SSIM 应使用同一套归一化参数；
   建议在原始强度（或固定全局缩放）上计算指标，避免每例分位数归一化掩盖差异。

6. **数据划分**：Day 2 起应固定 train/val/test 受试者列表并写入 `configs/`，
   避免同一受试者的切片同时出现在训练和测试中造成泄漏。

---

## 9. 真实数据检查结果摘要

| 文件 | Shape (raw) | Spacing (raw, mm) | Orientation (raw) | Canonical shape | Canonical spacing | 有效 3D | NaN/Inf |
|---|---|---|---|---|---|---|---|
| IXI002 | (256,256,150) | (0.9375,0.9375,1.2) | PSR | (150,256,256) | (1.2,0.9375,0.9375) | 是 | 无 |
| IXI012 | (256,256,150) | (0.9375,0.9375,1.2) | PSR | (150,256,256) | (1.2,0.9375,0.9375) | 是 | 无 |
| IXI013 | (256,256,150) | (0.9375,0.9375,1.2) | PSR | (150,256,256) | (1.2,0.9375,0.9375) | 是 | 无 |

数据类型：IXI002 为 `int16`，IXI012 / IXI013 为 `float64`。`normalize_mri` 统一
转为 `float32`，因此后续管线不受影响，但读取时应保留原始 dtype 记录。

强度统计（全体 / 前景 P1–P99）：

- IXI002：min/max = 0 / 1068；前景 P1=1，P50=154，P99=621
- IXI012：min/max = 0 / 3996.75；前景 P1=1.84，P50=154.22，P99=1733.09
- IXI013：min/max = 0 / 4466.22；前景 P1=1.94，P50=180.28，P99=1620.55

完整数值（affine、header、分位数等）见 `results/mri_inspection.json`。

---

## 10. Day 2：through-plane 退化 + Sinc 插值 baseline

### 10.1 退化算子与原论文的对应关系

SMORE Section II-A 把 MRI 采样写作 `y = S x + n`，Section III-A 的模拟退化是：

    HR -> 沿 through-plane 轴抗混叠低通 -> 按因子 r 下采样 -> LR（粗网格）

本实现（`src/degradation.py`）严格对应：

| 论文概念 | 本实现 |
|---|---|
| 层选激励的低通轮廓（anti-aliasing） | 沿 through-plane 轴在 k-space 乘低通 mask（`gaussian` 默认 / `box` / `fermi`） |
| 稀疏层采样（层间距 > 层内分辨率） | 沿该轴按 r 倍下采样（`slice(0, n, r)`） |
| 低采样网格 | `DegradedVolume.lr`（长度 `ceil(n/r)`，间距 ×r） |
| 数字恢复网格 | `sinc_interpolate` 零填充 + IFFT，恢复到 HR 网格 |

- `gaussian`：`H(f)=exp(-ln2·(f/f_c)²)`，半功率点 `f_c = Nyquist/r`，模拟层选激励的
  平滑抗混叠轮廓（默认，SMORE 采集模型）。
- `box`：理想矩形截断 `H=1{|f|≤f_c}`，等价于 Sinc 核；用于演示"采集与重建都假设
  理想带限"这一退化解（产生 Gibbs 振铃）。
- `fermi`：平滑过渡窗，抗振铃（对应论文标题的 anti-aliasing 思想）。

所有运算正确使用 `fftshift` / `ifftshift` 与频域索引，逆 FFT 取实部（虚部残差 ~1e-13）。

### 10.2 关键数据决策：退化轴 = R

Day 1 已确认这 3 例 IXI T1 是**矢状位存储**：canonical RAS+ 后 through-plane
（1.2 mm 厚层）方向是 **R（axis 0）**，而非通常的 S/Z。因此 Day 2 的退化沿
**canonical axis 0（R）**进行，而不是默认的 Z。代码用 `axis` 参数化，如需沿 S 做
对照实验，改 `--axis 2` 即可。

### 10.3 运行方式

```powershell
# 单元测试（合成数组，10 项）
python tests/test_degradation.py

# 真实数据端到端：退化 + Sinc baseline + 指标 + 图
python scripts/eval_interpolation.py --r 2 --axis 0 --filter gaussian
```

参数见 `configs/degradation.json`。

### 10.4 指标协议（统一，不逐方法调窗）

- **强度域**：HR 与重建结果共有的原始强度，`data_range` = HR 在 mask 内的
  `max-min`，对所有方法统一。
- **评价 mask**：`HR > 0.01·P99.9(HR)`，再做 3×3×3 二值腐蚀 ×2，排除颅骨与
  部分容积体素。
- **PSNR / SSIM**：在 mask 内计算（SSIM 用 3D 窗 win_size=7，在 mask 包围盒内）。

### 10.5 实际结果（r=2, gaussian 退化, Sinc 插值）

| 受试者 | PSNR (dB) | SSIM | MSE | LR shape | LR spacing (mm) |
|---|---|---|---:|---|---|
| IXI002 | 28.63 | 0.9231 | 1539.8 | (75,256,256) | (2.40, 0.9375, 0.9375) |
| IXI012 | 33.06 | 0.9484 | 7809.0 | (75,256,256) | (2.40, 0.9375, 0.9375) |
| IXI013 | 33.78 | 0.9491 | 8272.9 | (75,256,256) | (2.40, 0.9375, 0.9375) |

> IXI002 的 PSNR 偏低：该例为 `int16`（max≈1068），IXI012/013 为 `float64`
> （max≈4000+），`data_range` 差异导致 PSNR 不可跨例直接比较，但同一协议下
> 各例结果仍可作 baseline 参考。

### 10.6 已生成产物

- NIfTI：`data/processed/<subject>/` 下 `_HR.nii.gz`、`_LR_r2.nii.gz`、
  `_sinc_r2.nii.gz`（LR 的 affine 已按 r 更新方向向量）。
- 图：`results/figures/degradation/` 下每例的
  `_coronal_hr_sinc_error.png`、`_sagittal_hr_sinc_error.png`、
  `_kspace_spectrum.png`。
- 指标：`results/day2_metrics.json`。

### 10.7 已知局限

1. NIfTI 存的是**幅度图**，不是复数 k-space；对幅度图做 FFT 截断是**模拟**，
   不等同于真实少采相位编码线。
2. 退化沿 R 轴时放大倍数受 GT 层厚限制（1.2 mm → ×2/×3 较合理）。
3. 未做重采样到各向同性网格，避免引入额外插值误差。
4. 本阶段只有 Sinc 传统 baseline；尚未有学习型方法可对比。

---

## 11. Day 3：简化 SMORE(3D) 自监督训练

### 11.1 核心思想与轴映射

SMORE 的自监督关键是：**用低分辨率体数据 g 自身的高分辨率 in-plane 平面当训练目标**。
g 只在 through-plane（R）方向损失了分辨率，A-S 平面仍是高分辨率，因此：

- **训练**：取 g 的 A-S 切片（固定 R），沿 A（切片 axis 0）施加与 through-plane
  退化相同形式的一维模糊核 h(r)，得到输入 b = h(r)*g，目标 = 清晰的 A-S 切片。
  网络学习 b -> g，即"沿一个方向去模糊"。
- **推理**：把学到的 1D 去模糊能力应用到包含 R 方向的 R-A 切片（固定 S），
  R 是切片 axis 0，网络沿 R 去模糊，重组为三维。

两个平面的模糊方向都是 patch 的第 0 个空间轴，所以同一 2D 网络可以从 A 迁移到 R。

**为什么不需要外部 HR**：训练输入和目标都来自 g（Sinc 插值结果），HR 只在整个
推理结束后的最终评价阶段才被读取。

### 11.2 训练配置（`configs/smore_baseline.yaml`）

- 网络：EDSR，n_resblocks=8，n_feats=64，res_scale=0.1，单通道，无上采样，
  全局残差 `输出 = 输入 + 残差`，参数量 592,065。
- 数据：仅 `_sinc_r2.nii.gz`（g），patch 32×32，patches_per_slice=32，
  val_interval=5（每 5 个 R 切片留 1 个自监督验证）。
- 优化：Adam lr=1e-4，L1 loss，batch 16，max_steps 3000，FP32，
  num_workers=0（Windows）。
- 归一化：g 前景 P99.5 作为 scale，训练与推理共用，推理后乘回。

### 11.3 为什么 EDSR 不需要上采样层

Sinc 插值已经把 LR 放到了目标数字网格（与 HR 同尺寸）。网络的任务是**恢复被
低通模糊掉的细节**，而不是增加像素数量，因此输入输出空间尺寸相同，不需要
传统的 sub-pixel/转置卷积上采样模块。

### 11.4 实际结果（单受试者 IXI002，r=2）

| 方法 | PSNR (dB) | SSIM | MAE |
|---|---:|---:|---:|
| Sinc baseline | 28.631 | 0.9231 | 39.24 |
| 简化 SMORE(3D) | 28.901 | 0.9260 | 38.04 |
| **Δ** | **+0.269** | **+0.0029** | **-1.20** |

- 训练：3000 步，最佳自监督验证 L1=0.00148 @ step 2800，耗时 42s，
  峰值 GPU 显存 113 MB。
- 阶段 A（最小过拟合）通过：Loss 0.061→0.003，参数更新，梯度有限，无 NaN。
- 单元测试 8/8 通过（含"训练数据不读取 HR"的源码级检查）。

**提升较小是符合预期的**：Sinc 基线已经恢复了 Nyquist/r 以内的全部频率，SMORE
只能再恢复高斯模糊在带内的衰减（这部分很小）；被硬截断的高频（>Nyquist/r）
是信息论上不可恢复的。更大的提升需要更锋利的采集模糊核、更长训练、更大网络，
或原论文的多方向集成 / SAA 去混叠。

### 11.5 运行方式

```powershell
python tests/test_smore.py                # 单元测试（8 项）
python scripts/train_smore.py             # 阶段 A + B（先过拟合测试再完整训练）
python scripts/infer_smore.py             # 三维推理 -> _smore_r2.nii.gz
python scripts/eval_smore.py              # SMORE vs Sinc 指标 + 图
```

### 11.6 产物

- checkpoint：`results/checkpoints/smore_baseline/best.pt`、`last.pt`
- 日志：`results/logs/smore_baseline/training_summary.json`、`loss_history.json`
- 推理结果：`data/processed/IXI002-Guys-0828-T1/IXI002-Guys-0828-T1_smore_r2.nii.gz`
- 图：`results/figures/day3/`（训练样本、coronal/sagittal 对比与误差图）
- 指标：`results/day3_metrics.json`

### 11.7 与原论文完整 SMORE 的差别

1. **单方向推理**：只做 R-A（固定 S）一个方向；原论文还做多方向旋转集成。
2. **无 SAA（Super-Resolution with Anti-Aliasing）去混叠网络**。
3. **无 Fourier Burst Accumulation、无 Physics Consistency Loss、无 Gradient Loss**。
4. **网络更小**：8 残差块 / 64 通道（调试起始配置），非原论文完整超参数。
5. **单受试者**：仅 IXI002，未跨受试者训练。
6. **退化轴 = R**：本项目数据是矢状位存储，through-plane 方向是 R 而非 Z。

---

## 12. Day 4：Sobel Gradient Loss + 训练曲线 + 初步消融

### 12.1 损失定义

`src/losses.py` 实现 Sobel 梯度结构损失：

    L_total = L1(pred, target) + lambda_grad * L_grad(pred, target)
    L_grad  = mean( |Sobel_x(pred)-Sobel_x(target)| + |Sobel_y(pred)-Sobel_y(target)| )

- Sobel 核固定（register_buffer，不参与梯度更新），卷积本身可微，能反向传播。
- 用 `reflect` padding 避免边界补零伪影。
- `lambda_grad=0` 时 total 严格等于 L1（梯度项仍记录，但不加入）。
- **数量级校准**：实测 L1≈0.0059、梯度损失≈0.047（约为 L1 的 8 倍），故
  λ=0.01/0.05/0.10 分别对应梯度项约为 L1 的 8%/40%/80%。

### 12.2 记录系统

- TensorBoard：`results/tensorboard/<exp>/`（tag: train/val 的 total/l1/gradient + lr）
- CSV：`results/logs/csv/<exp>_train.csv`、`<exp>_val.csv`（字段含 global_step/epoch/
  各分项 loss/lr）
- PNG 曲线：`results/figures/training_curves/fig1..4*.png`（由 `scripts/plot_training_curves.py`
  读取真实 CSV 生成）

每个实验独立目录，互不覆盖；Day 3 的 `results/checkpoints/smore_baseline` 未被改动。

### 12.3 初步消融结果（单受试者 IXI002，r=2，3000 步，同预算/同种子）

| Method | PSNR (dB) | SSIM | MAE | GradMAE（边缘误差） |
|---|---:|---:|---:|---:|
| Sinc Interpolation | 28.631 | 0.9231 | 39.24 | 171.48 |
| SMORE + L1（λ=0） | 28.904 | 0.9259 | 38.03 | 136.42 |
| SMORE + L1+Grad（λ=0.01） | 28.909 | 0.9261 | 38.01 | 136.09 |
| SMORE + L1+Grad（λ=0.05） | 28.907 | 0.9262 | 38.02 | 136.10 |
| SMORE + L1+Grad（λ=0.10） | 28.904 | 0.9261 | 38.03 | 136.13 |

**结论（真实、非显著）**：

- SMORE（任何变体）相对 Sinc：PSNR +0.27 dB、**边缘误差 -20%**（171→136），
  说明学习型去模糊确实更好地保持了结构边界。
- 但 **Sobel 梯度损失在 λ∈{0.01,0.05,0.10} 下与纯 L1 几乎持平**（各指标差异
  在第三位小数），没有带来额外增益。这是诚实的负结果：Sobel 是线性算子，
  `L1 + λ·Sobel-L1` 与 L1 高度冗余，在当前预算下不改变最优解。
- 没有把"边缘更锐利"解释为"医学结构更真实"；也未据此宣称梯度损失能抑制幻觉。

### 12.4 运行方式

```powershell
python tests/test_losses.py                                   # 损失单元测试（6 项）
python scripts/train_smore.py --loss l1                       # baseline
python scripts/train_smore.py --loss l1_grad --lambda-grad 0.05
python scripts/plot_training_curves.py                        # 生成 4 张曲线图
python scripts/infer_smore.py --checkpoint results/checkpoints/day4_grad_005/best.pt \
    --output data/processed/IXI002-Guys-0828-T1/IXI002-Guys-0828-T1_smore_day4_grad005.nii.gz
python scripts/eval_smore.py                                  # 多方法对比 + 图
# TensorBoard 查看
tensorboard --logdir .\results\tensorboard --port 6006
```

### 12.5 产物

- checkpoint：`results/checkpoints/day4_l1/`、`day4_grad_001/`、`day4_grad_005/`、
  `day4_grad_010/`（各自 `best.pt`/`last.pt`）
- TensorBoard：`results/tensorboard/<exp>/events.out.tfevents.*`
- CSV：`results/logs/csv/<exp>_train.csv`、`_val.csv`、`_summary.json`
- 曲线：`results/figures/training_curves/fig1..4*.png`
- 重建：`data/processed/IXI002-Guys-0828-T1/*_smore_day4_*.nii.gz`
- 对比图：`results/figures/comparison/`
- 指标：`results/day4_metrics.json`
