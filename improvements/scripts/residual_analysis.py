"""保守残差分析（EXP009，零训练）：SR = Sinc + α·Residual。

研究问题：多退化训练是否主要通过"抑制错误锐化"改善鲁棒性（而非恢复更多细节）？
若固定模型在失配条件上过度锐化，则减小 α（收缩残差）应能改善其表现。

对固定模型与多退化模型，扫描 α ∈ {0, 0.25, 0.5, 0.75, 1.0}，报告 PSNR。

用法::

    python improvements/scripts/residual_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_mri
from src.edsr import EDSR
from src.metrics import brain_mask, psnr


def load_model(p, dev):
    ck = torch.load(p, map_location=dev); m = ck.get("config", {}).get("model", {})
    model = EDSR(m.get("n_resblocks", 8), m.get("n_feats", 64), m.get("res_scale", 0.1)).to(dev)
    model.load_state_dict(ck["model_state_dict"]); model.eval(); return model


def infer(model, g, dev, bs=4):
    n_r, n_a, n_s = g.shape; out = np.zeros_like(g, dtype=np.float32)
    with torch.no_grad():
        for s0 in range(0, n_s, bs):
            s1 = min(s0 + bs, n_s)
            batch = np.concatenate([g[:, :, s][None, None] for s in range(s0, s1)], 0)
            y = model(torch.from_numpy(batch).float().to(dev)).cpu().numpy()
            for k, s in enumerate(range(s0, s1)):
                out[:, :, s] = y[k, 0]
    return out


def main() -> int:
    dev = torch.device("cuda")
    s = "IXI002-Guys-0828-T1"; proc = Path("data/processed") / s
    hrv = load_mri(proc / f"{s}_HR.nii.gz", canonical=True); hr = hrv.data.astype(np.float64)
    mask = brain_mask(hr)

    cases = {
        "box": ("results/checkpoints/IXI002_pb_box_fixedg/best.pt",
                "results/checkpoints/IXI002_pb_box_multideg/best.pt"),
        "fermi": ("results/checkpoints/IXI002_pb_fermi_fixedg/best.pt",
                  "results/checkpoints/IXI002_pb_fermi_multideg/best.pt"),
    }
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    for flt, (fg_ckpt, md_ckpt) in cases.items():
        g_vol = load_mri(proc / f"{s}_sinc_r2_{flt}.nii.gz", canonical=True)
        g = g_vol.data.astype(np.float64)
        scale = float(np.percentile(g[g > 0], 99.5)) if (g > 0).any() else 1.0
        sinc = psnr(hr, g, mask)
        models = {"fixed-G": load_model(fg_ckpt, dev), "multi-D": load_model(md_ckpt, dev)}
        print(f"\n=== 实际退化={flt}，Sinc PSNR={sinc:.3f} ===")
        print(f"{'method':<10} " + "  ".join(f"a={a:.2f}" for a in alphas))
        for name, model in models.items():
            out = infer(model, (g / scale).astype(np.float32), dev) * scale
            residual = out - g  # 原始域残差
            row = [psnr(hr, g + a * residual, mask) for a in alphas]
            print(f"{name:<10} " + "  ".join(f"{v:>8.3f}" for v in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
