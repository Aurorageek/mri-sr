"""EXP018：可靠性引导残差恢复（RG-SMORE）与严格对照。

方法定义（全部基于同一 g 与同一 checkpoint 的 X_A / X_B）：
- Sinc: g
- Single-A / Single-B: X_A, X_B
- Average: (X_A+X_B)/2
- FBA: 频域 |F|^p 加权融合（p=3）
- Fixed-α: g + α (X_mean - g)
- Gradient gating: g + exp(-|∇g|/τ) (X_mean - g)
- RG-SMORE: g + exp(-U/τ) (X_mean - g)，U = |X_A - X_B| 平滑

关键对照：与 RG 具有相同残差能量的 Fixed-α（Energy-matched）。

用法::

    python improvements/scripts/reliability_guided.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_mri
from src.edsr import EDSR
from src.metrics import (
    brain_mask,
    gradient_3d_mae,
    mae,
    psnr,
    ssim,
    through_plane_gradient_mae,
)

SSIM_METHODS = {"Sinc", "Average", "FBA", "RG-tau0.5", "GradGate-tau0.5", "Fixed-a1"}

SUBJECTS = ["IXI002-Guys-0828-T1", "IXI012-HH-1211-T1",
            "IXI013-HH-1212-T1", "IXI022-Guys-0701-T1"]
DEV_SELECT = SUBJECTS[:2]
DEV_HOLDOUT = SUBJECTS[2:]
OUT = PROJECT_ROOT / "improvements" / "results" / "phase4"


def load_model(p, dev):
    ck = torch.load(p, map_location=dev); m = ck.get("config", {}).get("model", {})
    model = EDSR(m.get("n_resblocks", 8), m.get("n_feats", 64), m.get("res_scale", 0.1)).to(dev)
    model.load_state_dict(ck["model_state_dict"]); model.eval(); return model, ck


def infer_fix_axis(model, g, dev, fix_axis, bs=4):
    out = np.zeros_like(g, dtype=np.float32)
    n_slices = g.shape[fix_axis]
    with torch.no_grad():
        for i0 in range(0, n_slices, bs):
            i1 = min(i0 + bs, n_slices)
            batch = []
            for i in range(i0, i1):
                sl = [slice(None)] * 3; sl[fix_axis] = i
                batch.append(g[tuple(sl)][None, None, ...])
            x = torch.from_numpy(np.concatenate(batch, 0)).float().to(dev)
            y = model(x).cpu().numpy()
            for k, i in enumerate(range(i0, i1)):
                sl = [slice(None)] * 3; sl[fix_axis] = i
                out[tuple(sl)] = y[k, 0]
    return out


def cache_dir(s):
    d = PROJECT_ROOT / "data" / "processed" / s
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_views(s, dev):
    """返回 g, X_A, X_B（缓存到 npz，避免重复推理）。"""
    proc = PROJECT_ROOT / "data" / "processed" / s
    g = load_mri(proc / f"{s}_sinc_r2_paper.nii.gz", canonical=True).data.astype(np.float64)
    cache = proc / f"{s}_phase4_views.npz"
    if cache.exists():
        z = np.load(cache)
        X_A, X_B = z["X_A"].astype(np.float64), z["X_B"].astype(np.float64)
        if X_A.shape == g.shape:
            return g, X_A, X_B
    model, ck = load_model(PROJECT_ROOT / "results/checkpoints" / f"{s}_paperaligned" / "best.pt", dev)
    scale = float(ck.get("scale", np.percentile(g[g > 0], 99.5)))
    gn = (g / scale).astype(np.float32)
    X_A = infer_fix_axis(model, gn, dev, 2) * scale
    X_B = infer_fix_axis(model, gn, dev, 1) * scale
    np.savez_compressed(cache, X_A=X_A.astype(np.float32), X_B=X_B.astype(np.float32))
    return g, X_A.astype(np.float64), X_B.astype(np.float64)


def fba(a, b, p=3.0):
    A = np.fft.fftn(a); B = np.fft.fftn(b)
    wa = np.abs(A) ** p; wb = np.abs(B) ** p
    w = wa / (wa + wb + 1e-30)
    return np.real(np.fft.ifftn(w * A + (1 - w) * B))


def norm01(x, mask, q=95):
    """按 mask 内 Pq 归一化（只用 g 派生量，不用 HR）。"""
    ref = np.percentile(np.abs(x[mask]), q) + 1e-9
    return np.clip(np.abs(x) / ref, 0, 1)


def evaluate_all(s, dev, taus=(0.25, 0.5, 1.0), energy_match=True):
    proc = PROJECT_ROOT / "data" / "processed" / s
    hrv = load_mri(proc / f"{s}_HR.nii.gz", canonical=True)
    hr = hrv.data.astype(np.float64)
    spacing = tuple(float(v) for v in hrv.voxel_spacing)
    mask = brain_mask(hr)
    g, X_A, X_B = get_views(s, dev)
    X_mean = 0.5 * (X_A + X_B)

    U = ndimage.gaussian_filter(np.abs(X_A - X_B), 1.0)
    G = ndimage.generic_gradient_magnitude(g, ndimage.sobel)
    U_n = norm01(U, mask); G_n = norm01(G, mask)
    R = X_mean - g

    methods = {"Sinc": g, "Single-A": X_A, "Single-B": X_B,
               "Average": X_mean, "FBA": fba(X_A, X_B)}
    # 固定残差缩放
    for a in (0.25, 0.5, 0.75, 1.0):
        methods[f"Fixed-a{a:g}"] = g + a * R
    # 梯度门控（对照）
    for tau in taus:
        Wg = np.exp(-G_n / tau)
        methods[f"GradGate-tau{tau:g}"] = g + Wg * R
    # RG
    for tau in taus:
        Wu = np.exp(-U_n / tau)
        methods[f"RG-tau{tau:g}"] = g + Wu * R
        if energy_match:
            # 能量匹配的固定 alpha
            resid_energy = float(np.mean(((Wu * R)[mask]) ** 2))
            r_energy = float(np.mean((R[mask]) ** 2)) + 1e-30
            a_eq = float(np.sqrt(resid_energy / r_energy))
            methods[f"Fixed-aEQ(tau{tau:g},a={a_eq:.2f})"] = g + a_eq * R

    rows = []
    R_true = hr - g
    mse_sinc = float(np.mean((g[mask] - hr[mask]) ** 2))
    R_energy = float(np.mean(R[mask] ** 2)) + 1e-30
    for name, sr in methods.items():
        t0 = __import__("time").time()
        R_pred = sr - g
        mse_sr = float(np.mean((sr[mask] - hr[mask]) ** 2))
        pred_sq = float(np.mean(R_pred[mask] ** 2))
        cross = float(np.mean(R_pred[mask] * R_true[mask]))
        sp, sr_std = np.std(R_pred[mask]), np.std(R_true[mask])
        corr = float(np.corrcoef(R_pred[mask], R_true[mask])[0, 1]) if sp > 1e-12 and sr_std > 1e-12 else float("nan")
        harmful = float(np.mean((np.abs(sr[mask] - hr[mask]) > np.abs(g[mask] - hr[mask]) + 1.0)))
        row = {"method": name,
               "psnr_db": psnr(hr, sr, mask), "mae": mae(hr, sr, mask),
               "ssim": None, "tp_grad": through_plane_gradient_mae(hr, sr, mask, axis=0, spacing=spacing[0]),
               "grad3d": gradient_3d_mae(hr, sr, mask, spacing=spacing),
               "delta_mse": mse_sr - mse_sinc, "pred_sq": pred_sq, "cross_term": cross,
               "resid_corr": corr, "harmful_ratio": harmful,
               "energy_ratio": pred_sq / R_energy}
        if name in SSIM_METHODS:
            row["ssim"] = ssim(hr, sr, mask, win_size=7)
        rows.append(row)
        print(f"    {name:<30} done ({__import__('time').time()-t0:.1f}s)", flush=True)
    return rows


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for s in SUBJECTS:
        print(f"评估 {s} ...")
        rows = evaluate_all(s, dev)
        results[s] = rows
        print(f"  {'method':<26} {'PSNR':>8} {'SSIM':>8} {'TPGrad':>8} {'dMSE':>10} {'harm%':>7}")
        for r in rows:
            if r["method"] in ("Sinc", "Average", "FBA") or r["method"].startswith("RG") \
               or r["method"].startswith("Fixed-aEQ") or r["method"].startswith("GradGate"):
                ss = f"{r['ssim']:>8.5f}" if r["ssim"] is not None else f"{'--':>8}"
                print(f"  {r['method']:<26} {r['psnr_db']:>8.3f} {ss} "
                      f"{r['tp_grad']:>8.2f} {r['delta_mse']:>+10.1f} {100*r['harmful_ratio']:>6.2f}",
                      flush=True)
    (OUT / "reliability_guided_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {OUT / 'reliability_guided_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
