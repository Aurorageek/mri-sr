"""独立测试受试者正式验证（EXP015）。

对 4 名全新受试者（IXI022-025，未参与任何历史实验），在标准 Gaussian 退化（r=2）上
评价 Sinc / Fixed-G / Fixed-Box。

用法::

    python improvements/scripts/eval_independent.py
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
from src.metrics import brain_mask, evaluate_with_edge


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


def main() -> int:
    dev = torch.device("cuda")
    print(f"独立测试（4 名新受试者，标准 Gaussian 退化，r=2）\n")
    print(f"{'subject':<22} {'sinc':>8} {'Δ Fixed-G':>11} {'Δ Fixed-Box':>12}")
    rows = []
    for s in SUBJECTS:
        proc = PROJECT_ROOT / "data" / "processed" / s
        hr_vol = load_mri(proc / f"{s}_HR.nii.gz", canonical=True)
        hr = hr_vol.data.astype(np.float64)
        spacing = tuple(float(v) for v in hr_vol.voxel_spacing)
        mask = brain_mask(hr)
        g_vol = load_mri(proc / f"{s}_sinc_r2.nii.gz", canonical=True)
        g = g_vol.data.astype(np.float64)
        sinc = evaluate_with_edge(hr, g, mask, spacing=spacing)
        scale = float(np.percentile(g[g > 0], 99.5)) if (g > 0).any() else 1.0
        deltas = {}
        for name, tag in [("Fixed-G", "fixedg"), ("Fixed-Box", "fixedbox")]:
            model = load_model(PROJECT_ROOT / "results/checkpoints" / f"{s}_indep_{tag}" / "best.pt", dev)
            rec = infer(model, (g / scale).astype(np.float32), dev) * scale
            m = evaluate_with_edge(hr, rec.astype(np.float64), mask, spacing=spacing)
            deltas[name] = (m["psnr_db"] - sinc["psnr_db"], m["ssim"] - sinc["ssim"])
        rows.append((s, sinc, deltas))
        print(f"{s:<22} {sinc['psnr_db']:>8.3f} {deltas['Fixed-G'][0]:>+11.3f} {deltas['Fixed-Box'][0]:>+12.3f}")

    print("\n汇总（ΔPSNR vs Sinc）:")
    for name in ["Fixed-G", "Fixed-Box"]:
        ds = [r[2][name][0] for r in rows]
        print(f"  {name:<10} mean={np.mean(ds):+.3f}  worst={min(ds):+.3f}  (n={len(ds)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
