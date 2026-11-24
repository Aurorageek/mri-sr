"""Protocol B 评估（EXP007）：标准单受试者自监督适配下的退化失配。

对每种实际退化 D_actual ∈ {gaussian, box, fermi}：
- 生成/加载 g_{D_actual}（实际 LR）。
- 用 g_{D_actual} 自身训练（或加载已训练的）Fixed-Gaussian 与 Multi-Deg 模型。
- 在同一 g_{D_actual} 上推理，与 Sinc 基线、HR 比较。

回答：在正确的 Protocol B 下，退化失配是否仍导致性能下降？多退化是否仍有效？

用法::

    python improvements/scripts/eval_protocol_b.py
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
from src.metrics import brain_mask, evaluate_with_edge


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
    subject = "IXI002-Guys-0828-T1"
    proc = PROJECT_ROOT / "data" / "processed" / subject
    hr_vol = load_mri(proc / f"{subject}_HR.nii.gz", canonical=True)
    hr = hr_vol.data.astype(np.float64)
    spacing = tuple(float(v) for v in hr_vol.voxel_spacing)
    mask = brain_mask(hr)

    # Protocol B 模型映射：每种实际退化 -> 用该 g 训练的 Fixed-G 与 Multi-Deg
    model_map = {
        "gaussian": ("results/checkpoints/day4_l1/best.pt", "results/checkpoints/IXI002_multideg/best.pt"),
        "box": ("results/checkpoints/IXI002_pb_box_fixedg/best.pt", "results/checkpoints/IXI002_pb_box_multideg/best.pt"),
        "fermi": ("results/checkpoints/IXI002_pb_fermi_fixedg/best.pt", "results/checkpoints/IXI002_pb_fermi_multideg/best.pt"),
    }

    print(f"Protocol B 评估（{subject}，r=2）\n")
    print(f"{'actual':<10} {'sinc':>8} {'Δ fixed-G':>10} {'Δ multi-D':>10} "
          f"{'TP-Grad(fixed/multi)':>22}")

    for flt in ["gaussian", "box", "fermi"]:
        # 生成/加载 g
        g_path = proc / f"{subject}_sinc_r2_{flt}.nii.gz" if flt != "gaussian" else proc / f"{subject}_sinc_r2.nii.gz"
        g_vol = load_mri(g_path, canonical=True)
        g = g_vol.data.astype(np.float64)
        sinc = evaluate_with_edge(hr, g, mask, spacing=spacing)

        fg_ckpt, md_ckpt = model_map[flt]
        models = {
            "fixed-G": load_model(Path(fg_ckpt), device),
            "multi-D": load_model(Path(md_ckpt), device),
        }
        scale = float(np.percentile(g[g > 0], 99.5)) if (g > 0).any() else 1.0
        deltas = {}
        tp = {}
        for name, model in models.items():
            rec = infer(model, (g / scale).astype(np.float32), device) * scale
            m = evaluate_with_edge(hr, rec.astype(np.float64), mask, spacing=spacing)
            deltas[name] = m["psnr_db"] - sinc["psnr_db"]
            tp[name] = m["through_plane_gradient_mae"]
        print(f"{flt:<10} {sinc['psnr_db']:>8.3f} {deltas['fixed-G']:>+10.3f} "
              f"{deltas['multi-D']:>+10.3f} {tp['fixed-G']:.1f}/{tp['multi-D']:.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
