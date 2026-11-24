# Paper Experiments（英文论文实验素材）

> 所有数值均来自真实实验，来源标注在每节末尾。未完成项明确标注 TODO。

## Research Problem

Self-supervised MRI through-plane super-resolution (SMORE) constructs training pairs by
simulating anisotropic degradation from the subject's own in-plane data. This assumes the
simulated training degradation matches the actual LR input. When it does not, the network
can hallucinate high-frequency content and perform **worse than simple Sinc interpolation**.

## Motivation

- Day 4–6 established that SMORE (fixed Gaussian degradation) beats Sinc on the **matching**
  Gaussian condition (+0.32 dB, 8 subjects, p=0.008).
- EXP003/012 reveal that under **Box/Fermi** degradation the same model degrades by
  **1.2–1.8 dB below Sinc** — a reliability failure, not merely a smaller gain.
- This motivates a degradation-robust training recipe.

## Method Overview

Three self-supervised training variants (same EDSR, same budget, Protocol B):

| Variant | Training degradation |
|---|---|
| SMORE-Fixed-Gaussian | Gaussian low-pass |
| SMORE-Fixed-Box | Box (ideal truncation) low-pass |
| SMORE-Uniform-Multi | random {Gaussian, Box, Fermi} |

Key finding: **Box-trained SMORE is intrinsically robust**; it does not require
multi-degradation sampling.

## Experimental Setup

- Data: IXI-T1, 10 development + 4 independent test subjects (IXI022–025).
- Degradation: r=2, through-plane axis = R (canonical axis 0), voxel spacing
  1.2×0.9375×0.9375 mm.
- Network: simplified EDSR (8 residual blocks, 64 features, no BN, no upsampling,
  global residual), 592,065 params.
- Training: L1, Adam 1e-4, batch 16, 3000 steps, seed 42, FP32, 113 MB peak GPU.
- Evaluation: foreground mask (HR > 0.01·P99.9, erosion ×2); PSNR with HR foreground
  data range; SSIM (3D, win 7); MAE; through-plane gradient MAE; 3D gradient MAE.

## Main Results

### Full degradation matrix (development subject IXI002, ΔPSNR vs Sinc, dB)

| Actual \ Training | Fixed-G | Fixed-Box | Fixed-Fermi | Uniform-Multi |
|---|---:|---:|---:|---:|
| Gaussian | +0.273 | +0.222 | +0.224 | +0.412 |
| Box | **-1.637** | +0.340 | +0.192 | +0.001 |
| Fermi | **-1.236** | +0.291 | +0.248 | +0.100 |
| mean | -0.867 | **+0.284** | +0.221 | +0.171 |

### Independent test (4 unseen subjects, 12 subject×condition cells)

| Method | mean ΔPSNR | worst | positive cells |
|---|---:|---:|---:|
| Fixed-Gaussian | -0.904 | -1.811 | 4/12 |
| Fixed-Box | **+0.244** | **+0.062** | **12/12** |

## Ablation Study

1. **Multi vs single**: Fixed-Box (+0.284 mean) > Uniform-Multi (+0.171) → uniform
   sampling is unnecessary.
2. **Degradation estimate feasibility**: spectra of Gaussian/Box/Fermi LR are
   indistinguishable (corr > 0.99999) → adaptive sampling has no reliable signal.
3. **Residual scaling**: Fixed-G residual is monotonically harmful under mismatch;
   Fixed-Box residual is neutral-to-positive.

## Robustness Analysis

- Matching condition: Fixed-G +0.26, Fixed-Box +0.17 (both > Sinc).
- Mismatch condition: Fixed-G -0.90 to -1.81; Fixed-Box +0.06 to +0.37.
- **Fixed-Box never falls below Sinc in any of 12 independent cells.**

## Structural Fidelity Analysis

- Through-plane gradient MAE: Fixed-G raises it under Box (pseudo-edges from
  over-sharpening); Fixed-Box stays close to Sinc.
- No visible over-sharpening or spurious texture in Fixed-Box reconstructions
  (see `figures/paper/fig5_*`).

## Limitations

- 4 independent subjects (expand to 8–10).
- Only r=2 shows a clear benefit; at r=4 the gain nearly vanishes (+0.005 dB).
- Degradation simulation is on magnitude images, not complex k-space.
- The mechanism ("Box is more transferable") remains a hypothesis.

## Conclusion

Training SMORE with a **Box** low-pass kernel, rather than the conventional Gaussian,
yields a degradation-robust self-supervised MRI SR model that consistently improves over
Sinc interpolation across Gaussian/Box/Fermi conditions, with no extra computation.

---

## Sources

- Matrix: `improvements/experiments/EXP012/analysis.md`
- Fixed-Box cross-subject: `improvements/experiments/EXP013/analysis.md`
- Feature feasibility: `improvements/experiments/EXP014/analysis.md`
- Independent test: `improvements/experiments/EXP015/analysis.md`
- Figures: `improvements/figures/paper/`
