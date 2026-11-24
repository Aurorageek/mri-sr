"""方向 D：多方向重建融合（EXP002，复用 checkpoint，无需训练）。

研究问题：单方向推理（R-A，fix S）是否足够？同时用 R-A 与 R-S 两个正交方向
推理并融合，能否降低重建误差？

- 单方向 A：对 R-A 切片（fix S）沿 R 去模糊 -> vol_A（即现有 SMORE 推理）。
- 单方向 S：对 R-S 切片（fix A）沿 R 去模糊 -> vol_S。
- 融合：vol_fused = (vol_A + vol_S) / 2。

两个方向的待去模糊轴都是 R（patch 第 0 轴），只是横向上下文不同（A vs S）。

用法::

    python improvements/scripts/multiview_infer.py --subject IXI002-Guys-0828-T1 \
        --checkpoint results/checkpoints/day4_l1/best.pt
"""

from __future__ import annotations

import argparse
import json
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


def load_model(ckpt_path: Path, device: torch.device) -> tuple[EDSR, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    m = ckpt.get("config", {}).get("model", {})
    model = EDSR(m.get("n_resblocks", 8), m.get("n_feats", 64), m.get("res_scale", 0.1),
                 m.get("n_channels", 1)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def infer_fix_axis(model, g, device, fix_axis, batch_size=4):
    """固定 fix_axis（1=A 或 2=S），逐层沿 R（axis0）去模糊。返回体数据。"""
    n_r, n_a, n_s = g.shape
    out = np.zeros_like(g, dtype=np.float32)
    n_slices = g.shape[fix_axis]
    with torch.no_grad():
        for i0 in range(0, n_slices, batch_size):
            i1 = min(i0 + batch_size, n_slices)
            batch = []
            for i in range(i0, i1):
                sl = [slice(None)] * 3
                sl[fix_axis] = i
                slc = g[tuple(sl)]  # (R, X)，R 为 axis0
                batch.append(slc[None, None, ...])
            x = torch.from_numpy(np.concatenate(batch, axis=0)).float().to(device)
            y = model(x).cpu().numpy()
            for k, i in enumerate(range(i0, i1)):
                sl = [slice(None)] * 3
                sl[fix_axis] = i
                out[tuple(sl)] = y[k, 0]
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="多方向重建融合")
    p.add_argument("--subject", default="IXI002-Guys-0828-T1")
    p.add_argument("--checkpoint", default=str(PROJECT_ROOT / "results/checkpoints/day4_l1/best.pt"))
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    proc = PROJECT_ROOT / "data" / "processed" / args.subject
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g_vol = load_mri(proc / f"{args.subject}_sinc_r2.nii.gz", canonical=True)
    hr = load_mri(proc / f"{args.subject}_HR.nii.gz", canonical=True).data.astype(np.float64)
    spacing = tuple(float(v) for v in g_vol.voxel_spacing)
    g = g_vol.data.astype(np.float64)
    scale = float(np.percentile(g[g > 0], 99.5)) if (g > 0).any() else 1.0
    g_norm = (g / scale).astype(np.float32)

    model, ckpt = load_model(Path(args.checkpoint), device)
    print(f"checkpoint scale={ckpt.get('scale'):.4f}, 使用本地 g 的 scale={scale:.4f}")

    # 单方向 R-A（fix S=2）与 R-S（fix A=1）
    vol_A = infer_fix_axis(model, g_norm, device, fix_axis=2, batch_size=args.batch_size) * scale
    vol_S = infer_fix_axis(model, g_norm, device, fix_axis=1, batch_size=args.batch_size) * scale
    fused = (vol_A + vol_S) / 2.0

    mask = brain_mask(hr)
    names = {"single_RA": vol_A, "single_RS": vol_S, "fused_avg": fused}
    results = {}
    for name, v in names.items():
        results[name] = evaluate_with_edge(hr, v.astype(np.float64), mask,
                                           spacing=spacing, through_plane_axis=0)

    print(f"\n{args.subject} 多方向融合对比:")
    for name, m in results.items():
        print(f"  {name:<12} PSNR={m['psnr_db']:.3f}  SSIM={m['ssim']:.5f}  MAE={m['mae']:.3f}  "
              f"TP-Grad={m['through_plane_gradient_mae']:.2f}  3D-Grad={m['gradient_3d_mae']:.2f}")

    # 保存融合结果 NIfTI
    import nibabel as nib
    out = proc / f"{args.subject}_smore_fused.nii.gz"
    nib.save(nib.Nifti1Image(fused.astype(np.float32), g_vol.affine), str(out))
    print(f"\n融合结果已保存: {out}")

    if args.output_json:
        payload = {k: {kk: (vv if not isinstance(vv, float) or np.isfinite(vv) else None)
                       for kk, vv in v.items()} for k, v in results.items()}
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
