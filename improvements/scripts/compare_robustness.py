"""汇总多退化训练 vs 固定高斯训练在三种退化条件下的鲁棒性（EXP005）。

对每个受试者，比较 Gaussian 模型与 Multi-Deg 模型在 Gaussian/Box/Fermi 三种
实际退化下的 PSNR，并给出相对各自 Sinc 的 ΔPSNR。

用法::

    python improvements/scripts/compare_robustness.py
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
from src.degradation import degrade_through_plane, sinc_interpolate
from src.edsr import EDSR
from src.metrics import brain_mask, psnr


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    m = ckpt.get("config", {}).get("model", {})
    model = EDSR(m.get("n_resblocks", 8), m.get("n_feats", 64), m.get("res_scale", 0.1)).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    return model


def infer(model, g, device, batch_size=4):
    n_r, n_a, n_s = g.shape
    out = np.zeros_like(g, dtype=np.float32)
    with torch.no_grad():
        for s0 in range(0, n_s, batch_size):
            s1 = min(s0 + batch_size, n_s)
            batch = np.concatenate([g[:, :, s][None, None] for s in range(s0, s1)], axis=0)
            y = model(torch.from_numpy(batch).float().to(device)).cpu().numpy()
            for k, s in enumerate(range(s0, s1)):
                out[:, :, s] = y[k, 0]
    return out


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subjects = {
        "IXI002-Guys-0828-T1": ("results/checkpoints/day4_l1/best.pt", "IXI002_multideg"),
        "IXI012-HH-1211-T1": ("results/checkpoints/dev_IXI012_l1/best.pt", "IXI012-HH-1211-T1_multideg"),
        "IXI013-HH-1212-T1": ("results/checkpoints/IXI013-HH-1212-T1_l1/best.pt", "IXI013-HH-1212-T1_multideg"),
        "IXI016-Guys-0697-T1": ("results/checkpoints/IXI016-Guys-0697-T1_l1/best.pt", "IXI016-Guys-0697-T1_multideg"),
    }
    filters = ["gaussian", "box", "fermi"]

    rows = []
    for s, (gauss_ckpt, md_name) in subjects.items():
        proc = PROJECT_ROOT / "data" / "processed" / s
        hr_vol = load_mri(proc / f"{s}_HR.nii.gz", canonical=True)
        hr = hr_vol.data.astype(np.float64)
        mask = brain_mask(hr)
        md_ckpt = PROJECT_ROOT / "results/checkpoints" / md_name / "best.pt"

        models = {"gaussian": load_model(Path(gauss_ckpt), device),
                  "multideg": load_model(md_ckpt, device)}

        print(f"\n=== {s} ===")
        print(f"{'actual':<10} {'sinc':>8} {'Δ gauss':>9} {'Δ multideg':>11}")
        for flt in filters:
            deg = degrade_through_plane(hr, axis=0, r=2.0, filter_name=flt)
            g = sinc_interpolate(deg.lr, axis=0, r=2.0, target_n=hr.shape[0])
            psinc = psnr(hr, g, mask)
            scale = float(np.percentile(g[g > 0], 99.5)) if (g > 0).any() else 1.0
            deltas = {}
            for name, model in models.items():
                rec = infer(model, (g / scale).astype(np.float32), device) * scale
                deltas[name] = psnr(hr, rec.astype(np.float64), mask) - psinc
            print(f"{flt:<10} {psinc:>8.3f} {deltas['gaussian']:>+9.3f} {deltas['multideg']:>+11.3f}")
            rows.append({"subject": s, "actual": flt, "sinc": round(psinc, 3),
                         "delta_gauss": round(deltas["gaussian"], 3),
                         "delta_multideg": round(deltas["multideg"], 3)})

    # 汇总最差情况（非 gaussian 条件下 Δ 的最小值）
    print("\n=== 汇总：非匹配条件下的 ΔPSNR（越小越差） ===")
    for s in subjects:
        mm = [r for r in rows if r["subject"] == s and r["actual"] != "gaussian"]
        g_min = min(r["delta_gauss"] for r in mm)
        m_min = min(r["delta_multideg"] for r in mm)
        print(f"{s:<22} gauss 最差 Δ={g_min:+.2f}  multideg 最差 Δ={m_min:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
