"""完整退化失配矩阵评估（EXP012）：3 实际退化 × 5 方法。

方法：Sinc, Fixed-Gaussian, Fixed-Box, Fixed-Fermi, Uniform-Multi。
每个学习方法都在对应实际 g 上自监督训练（Protocol B）。
输出相对 Sinc 的 ΔPSNR 矩阵，对角线即 Oracle-Matched。

用法::

    python improvements/scripts/eval_full_matrix.py
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


# 每种实际退化 -> 各训练方法的 checkpoint（Protocol B）
CKPT = {
    "gaussian": {"Fixed-G": "day4_l1", "Fixed-Box": "IXI002_fixedbox_on_gaussian",
                 "Fixed-Fermi": "IXI002_fixedfermi_on_gaussian", "Multi": "IXI002_multideg"},
    "box": {"Fixed-G": "IXI002_pb_box_fixedg", "Fixed-Box": "IXI002_fixedbox_on_box",
            "Fixed-Fermi": "IXI002_fixedfermi_on_box", "Multi": "IXI002_pb_box_multideg"},
    "fermi": {"Fixed-G": "IXI002_pb_fermi_fixedg", "Fixed-Box": "IXI002_fixedbox_on_fermi",
              "Fixed-Fermi": "IXI002_fixedfermi_on_fermi", "Multi": "IXI002_pb_fermi_multideg"},
}


def main() -> int:
    dev = torch.device("cuda")
    s = "IXI002-Guys-0828-T1"; proc = PROJECT_ROOT / "data" / "processed" / s
    hr_vol = load_mri(proc / f"{s}_HR.nii.gz", canonical=True)
    hr = hr_vol.data.astype(np.float64)
    mask = brain_mask(hr)

    methods = ["Fixed-G", "Fixed-Box", "Fixed-Fermi", "Multi"]
    print(f"退化失配矩阵（{s}，r=2，ΔPSNR vs Sinc）\n")
    print(f"{'actual':<10} {'sinc':>8} " + "".join(f"{m:>12}" for m in methods))

    matrix = {}
    for actual in ["gaussian", "box", "fermi"]:
        g_name = f"{s}_sinc_r2.nii.gz" if actual == "gaussian" else f"{s}_sinc_r2_{actual}.nii.gz"
        g_vol = load_mri(proc / g_name, canonical=True)
        g = g_vol.data.astype(np.float64)
        sinc = psnr(hr, g, mask)
        scale = float(np.percentile(g[g > 0], 99.5)) if (g > 0).any() else 1.0
        deltas = {}
        for m in methods:
            ckpt = PROJECT_ROOT / "results/checkpoints" / CKPT[actual][m] / "best.pt"
            model = load_model(ckpt, dev)
            rec = infer(model, (g / scale).astype(np.float32), dev) * scale
            deltas[m] = psnr(hr, rec.astype(np.float64), mask) - sinc
        matrix[actual] = (sinc, deltas)
        print(f"{actual:<10} {sinc:>8.3f} " + "".join(f"{deltas[m]:>+12.3f}" for m in methods))

    # 汇总：各方法的平均/最差 ΔPSNR
    print("\n=== 汇总（ΔPSNR vs Sinc，3 条件） ===")
    for m in methods:
        ds = [matrix[a][1][m] for a in matrix]
        print(f"{m:<12} mean={np.mean(ds):+.3f}  worst={min(ds):+.3f}  best={max(ds):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
