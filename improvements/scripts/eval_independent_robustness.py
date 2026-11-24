"""独立测试受试者的退化稳健性矩阵（EXP015 结果 2）。

4 名全新受试者 × 3 种实际退化 × {Fixed-G, Fixed-Box}（Protocol B），
报告 ΔPSNR vs Sinc，验证 Fixed-Box 的稳健性是否在独立数据上成立。

用法::

    python improvements/scripts/eval_independent_robustness.py
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


SUBJECTS = ["IXI022-Guys-0701-T1", "IXI023-Guys-0699-T1",
            "IXI024-Guys-0705-T1", "IXI025-Guys-0852-T1"]


def ckpt_name(s, actual, method):
    """检查点目录命名：独立测试用 <s>_indep[_<actual>]_<fixedg|fixedbox>。"""
    if method == "Fixed-G":
        return f"{s}_indep_fixedg" if actual == "gaussian" else f"{s}_indep_{actual}_fixedg"
    else:
        return f"{s}_indep_fixedbox" if actual == "gaussian" else f"{s}_indep_{actual}_fixedbox"


def main() -> int:
    dev = torch.device("cuda")
    print(f"{'subject':<22} {'sinc':>8} | {'G:gauss':>8} {'G:box':>8} {'G:fermi':>8} | "
          f"{'B:gauss':>8} {'B:box':>8} {'B:fermi':>8}")
    summary = {"Fixed-G": [], "Fixed-Box": []}
    for s in SUBJECTS:
        proc = PROJECT_ROOT / "data" / "processed" / s
        hr_vol = load_mri(proc / f"{s}_HR.nii.gz", canonical=True)
        hr = hr_vol.data.astype(np.float64)
        mask = brain_mask(hr)
        deltas = {"Fixed-G": {}, "Fixed-Box": {}}
        sinc_psnr = None
        for actual in ["gaussian", "box", "fermi"]:
            g_name = f"{s}_sinc_r2.nii.gz" if actual == "gaussian" else f"{s}_sinc_r2_{actual}.nii.gz"
            g_vol = load_mri(proc / g_name, canonical=True)
            g = g_vol.data.astype(np.float64)
            sinc = psnr(hr, g, mask)
            if actual == "gaussian":
                sinc_psnr = sinc
            scale = float(np.percentile(g[g > 0], 99.5)) if (g > 0).any() else 1.0
            for method in ["Fixed-G", "Fixed-Box"]:
                ckpt = PROJECT_ROOT / "results/checkpoints" / ckpt_name(s, actual, method) / "best.pt"
                model = load_model(ckpt, dev)
                rec = infer(model, (g / scale).astype(np.float32), dev) * scale
                deltas[method][actual] = psnr(hr, rec.astype(np.float64), mask) - sinc
                summary[method].append(deltas[method][actual])
        print(f"{s:<22} {sinc_psnr:>8.3f} | "
              + " ".join(f"{deltas['Fixed-G'][a]:>+8.3f}" for a in ["gaussian", "box", "fermi"])
              + " | " + " ".join(f"{deltas['Fixed-Box'][a]:>+8.3f}" for a in ["gaussian", "box", "fermi"]))

    print("\n汇总（n=12 受试者×条件）:")
    for m in ["Fixed-G", "Fixed-Box"]:
        ds = summary[m]
        print(f"  {m:<10} mean={np.mean(ds):+.3f}  worst={min(ds):+.3f}  "
              f"n_positive={sum(1 for d in ds if d > 0)}/{len(ds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
