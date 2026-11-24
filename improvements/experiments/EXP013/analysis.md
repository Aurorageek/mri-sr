# EXP013：Fixed-Box 跨受试者稳健性（最终方法验证）

- **Research Question**：Fixed-Box 的稳健性是否跨受试者成立？
- **Method**：4 名受试者 × 3 种实际退化，各训练 Fixed-Box（Protocol B），评价 ΔPSNR vs Sinc。
- **Results**：

| subject | gauss | box | fermi | mean | worst |
|---|---:|---:|---:|---:|---:|
| IXI002 | +0.222 | +0.340 | +0.291 | +0.284 | +0.222 |
| IXI012 | +0.195 | +0.313 | +0.192 | +0.234 | +0.192 |
| IXI013 | +0.187 | +0.346 | +0.311 | +0.281 | +0.187 |
| IXI016 | +0.140 | +0.200 | +0.190 | +0.177 | +0.140 |

- **跨受试者平均 ΔPSNR = +0.244 dB；最差情况 = +0.140 dB（全部为正，从不低于 Sinc）。**
- **Conclusion**：Fixed-Box 在所有受试者、所有退化条件下稳健优于 Sinc，无需多退化或
  自适应。**最终方法冻结为 SMORE-Fixed-Box**。
