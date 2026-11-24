# EXP014：退化特征可行性分析（退化估计不可行）

- **Research Question**：能否仅凭 LR MRI 频谱特征区分 Gaussian/Box/Fermi 退化？
- **Method**：比较三种 g 沿 through-plane（R）方向的归一化功率谱及其相关性。
- **Results**：三种退化的功率谱相关性 > 0.99999（几乎完全一致）。
- **Conclusion**：**三种退化在频域几乎不可区分**（截止频率相同、仅过渡形状不同）。
  因此基于 LR 频谱的退化估计**不可行**，自适应退化采样方法缺乏可靠依据，应放弃。
  这也进一步支持 Fixed-Box 这一无需退化估计的简单方案。
- **Decision**：DISCARD 自适应退化估计方向；确认 Fixed-Box 为最终方法。
