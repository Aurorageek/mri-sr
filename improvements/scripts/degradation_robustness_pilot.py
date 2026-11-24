"""方向 B：退化模型失配的零训练预实验（EXP003）。

用已训练好的 Gaussian 模型，在 Gaussian/Box/Fermi 三种滤波器生成的 LR 上推理，
看滤波器失配（训练 Gaussian、测试 Box/Fermi）是否影响重建。

零新训练，只复用 results/checkpoints/day4_l1/best.pt。

用法::

    python improvements/scripts/degradation_robustness_pilot.py --subject IXI002-Guys-0828-T1
"""

from __future__ import annotations

import argparse
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
    return model, ckpt


def infer_volume(model, g, device, batch_size=4):
    n_r, n_a, n_s = g.shape
    out = np.zeros_like(g, dtype=np.float32)
    with torch.no_grad():
        for s0 in range(0, n_s, batch_size):
            s1 = min(s0 + batch_size, n_s)
            batch = np.concatenate([g[:, :, s][None, None] for s in range(s0, s1)], axis=0)
            x = torch.from_numpy(batch).float().to(device)
            y = model(x).cpu().numpy()
            for k, s in enumerate(range(s0, s1)):
                out[:, :, s] = y[k, 0]
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--subject", default="IXI002-Guys-0828-T1")
    p.add_argument("--checkpoint", default=str(PROJECT_ROOT / "results/checkpoints/day4_l1/best.pt"))
    p.add_argument("--r", type=float, default=2.0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    proc = PROJECT_ROOT / "data" / "processed" / args.subject
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hr_vol = load_mri(proc / f"{args.subject}_HR.nii.gz", canonical=True)
    hr = hr_vol.data.astype(np.float64)
    spacing = tuple(float(v) for v in hr_vol.voxel_spacing)
    mask = brain_mask(hr)

    model, _ = load_model(Path(args.checkpoint), device)

    print(f"{args.subject} 退化失配预实验（模型=Gaussian 训练，r={args.r}）\n")
    print(f"{'filter':<10} {'PSNR':>8} {'SSIM':>8} {'MAE':>9} {'TP-Grad':>9} {'3D-Grad':>9}")

    for flt in ["gaussian", "box", "fermi"]:
        # 用该滤波器生成实际 LR（低通+下采样+Sinc 插值）
        deg = degrade_through_plane(hr, axis=0, r=args.r, filter_name=flt)
        g = sinc_interpolate(deg.lr, axis=0, r=args.r, target_n=hr.shape[0])
        # Sinc baseline（该条件下的直接插值）
        base = evaluate_with_edge(hr, g, mask, spacing=spacing)
        # 用 Gaussian 模型推理
        scale = float(np.percentile(g[g > 0], 99.5)) if (g > 0).any() else 1.0
        deblurred = infer_volume(model, (g / scale).astype(np.float32), device) * scale
        rec = evaluate_with_edge(hr, deblurred.astype(np.float64), mask, spacing=spacing)
        print(f"{flt:<10} sinc {base['psnr_db']:>7.3f} {base['ssim']:>8.5f} {base['mae']:>9.3f} "
              f"{base['through_plane_gradient_mae']:>9.2f} {base['gradient_3d_mae']:>9.2f}")
        print(f"{'':<10} smore {rec['psnr_db']:>7.3f} {rec['ssim']:>8.5f} {rec['mae']:>9.3f} "
              f"{rec['through_plane_gradient_mae']:>9.2f} {rec['gradient_3d_mae']:>9.2f}")

    print("\n解读：若 smore 在 box/fermi 条件相比 gaussian 条件明显下降，则存在退化失配。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
