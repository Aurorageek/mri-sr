"""Fixed-Box 跨受试者稳健性验证（EXP013）。

对每个受试者，在 3 种实际退化上各训练 Fixed-Box（Protocol B），评价其
ΔPSNR vs Sinc，报告 mean/worst，确认 Fixed-Box 是否跨条件、跨受试者稳健。

用法::

    python improvements/scripts/eval_fixedbox_robustness.py
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


SUBJECTS = {
    "IXI002-Guys-0828-T1": "IXI002",
    "IXI012-HH-1211-T1": "IXI012-HH-1211-T1",
    "IXI013-HH-1212-T1": "IXI013-HH-1212-T1",
    "IXI016-Guys-0697-T1": "IXI016-Guys-0697-T1",
}


def main() -> int:
    dev = torch.device("cuda")
    print(f"{'subject':<22} {'gauss':>8} {'box':>8} {'fermi':>8} {'mean':>8} {'worst':>8}")
    all_mean = []
    for s, short in SUBJECTS.items():
        proc = PROJECT_ROOT / "data" / "processed" / s
        hr_vol = load_mri(proc / f"{s}_HR.nii.gz", canonical=True)
        hr = hr_vol.data.astype(np.float64)
        mask = brain_mask(hr)
        deltas = []
        for actual in ["gaussian", "box", "fermi"]:
            g_name = f"{s}_sinc_r2.nii.gz" if actual == "gaussian" else f"{s}_sinc_r2_{actual}.nii.gz"
            g_vol = load_mri(proc / g_name, canonical=True)
            g = g_vol.data.astype(np.float64)
            sinc = psnr(hr, g, mask)
            scale = float(np.percentile(g[g > 0], 99.5)) if (g > 0).any() else 1.0
            ckpt = PROJECT_ROOT / "results/checkpoints" / f"{short}_fixedbox_on_{actual}" / "best.pt"
            model = load_model(ckpt, dev)
            rec = infer(model, (g / scale).astype(np.float32), dev) * scale
            deltas.append(psnr(hr, rec.astype(np.float64), mask) - sinc)
        all_mean.append(np.mean(deltas))
        print(f"{s:<22} " + "".join(f"{d:>+8.3f}" for d in deltas)
              + f"{np.mean(deltas):>+8.3f}{min(deltas):>+8.3f}")
    print(f"\n跨受试者 Fixed-Box 平均 ΔPSNR = {np.mean(all_mean):+.3f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
