# EXP001：退化算子对齐的数值分析

- **Research Question**：训练退化（in-plane 单轴高斯低通）与实际退化（高斯低通+下采样+Sinc）是否等价？
- **Hypothesis**：两者在频域不等价（实际退化含 box 截断与混叠），可能限制 SMORE 重建性能。
- **Method**：1D 合成信号 + 真实 MRI R 轴剖面，比较 D_train=gaussian(x) 与 D_real=sinc(downsample(gaussian(x))) 的频谱与相关性。
- **Results**：
  - 带限信号（f<0.15）：corr=1.0，两者等价。
  - 含高频信号（f<0.45）：corr=0.87，D_real 截断高频，两者不等价。
  - 真实 MRI（3 条 R 剖面）：D_train 高频(>0.25)残留仅 0.03–0.43%，corr=0.990–0.999。
- **Conclusion**：对真实 MRI，高斯滤波（f_cut=0.25）已几乎移除 >0.25 高频，额外下采样+Sinc 的差异可忽略 → 训练退化与实际退化近似等价。
- **Decision**：**DISCARD（无需训练对齐退化）**。假设被证伪：退化失配不显著，不值得消耗算力训练。
- **Note**：仍存在轻微物理尺度失配（A 体素 0.9375mm vs R 体素 1.2mm，f_cut=0.25 cycles/voxel 对应的物理截止频率不同），但属二阶效应。
- **代码**：`improvements/scripts/analyze_degradation_alignment.py`。
