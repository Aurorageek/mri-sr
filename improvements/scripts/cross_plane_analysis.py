"""EXP017：跨平面预测分歧与重建误差的关系分析（无需新训练）。

对每个开发受试者：
1. 用同一个 paper-aligned checkpoint 分别沿 R-A（fix S）与 R-S（fix A）推理，
   得到 X_A、X_B（同一物理网格）。
2. 构造分歧图 U = |X_A - X_B|（并做局部平滑）。
3. 与 HR 比较，得到 E_A、E_B、E_mean，以及相对 Sinc 的有害修正
   H = max(0, E_mean - E_sinc)。
4. 统计 U 与 E_mean / H 的 Spearman 相关，并用简单控制量（Sinc 梯度幅值、
   局部方差、残差幅值）做偏相关，判断 U 是否提供额外信息。
5. 按 U 分位数分组，比较各组误差。
6. 输出 JSON 与图表。

用法::

    python improvements/scripts/cross_plane_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import load_mri
from src.edsr import EDSR
from src.metrics import brain_mask

SUBJECTS = ["IXI002-Guys-0828-T1", "IXI012-HH-1211-T1",
            "IXI013-HH-1212-T1", "IXI022-Guys-0701-T1"]
OUT = PROJECT_ROOT / "improvements" / "results" / "phase4"
FIG = PROJECT_ROOT / "improvements" / "figures" / "phase4"


def load_model(p, dev):
    ck = torch.load(p, map_location=dev); m = ck.get("config", {}).get("model", {})
    model = EDSR(m.get("n_resblocks", 8), m.get("n_feats", 64), m.get("res_scale", 0.1)).to(dev)
    model.load_state_dict(ck["model_state_dict"]); model.eval(); return model, ck


def infer_fix_axis(model, g, dev, fix_axis, bs=4):
    """固定 fix_axis（1=A -> R-S 切片；2=S -> R-A 切片），沿 R（axis0）去模糊。"""
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


def rank_partial_spearman(a, b, c, mask):
    """rank(a) 与 rank(b) 在以 rank(c) 为控制变量后的偏 Spearman 相关。"""
    from scipy.stats import rankdata
    x = a[mask]; y = b[mask]; z = c[mask]
    Rx, Ry, Rz = rankdata(x), rankdata(y), rankdata(z)
    def residual(v):
        A = np.vstack([np.ones_like(Rz), Rz]).T
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ coef
    ex, ey = residual(Rx), residual(Ry)
    r = np.corrcoef(ex, ey)[0, 1]
    return float(r)


def analyze_subject(s, dev):
    proc = PROJECT_ROOT / "data" / "processed" / s
    hrv = load_mri(proc / f"{s}_HR.nii.gz", canonical=True)
    hr = hrv.data.astype(np.float64)
    g = load_mri(proc / f"{s}_sinc_r2_paper.nii.gz", canonical=True).data.astype(np.float64)
    mask = brain_mask(hr)

    model, ck = load_model(PROJECT_ROOT / "results/checkpoints" / f"{s}_paperaligned" / "best.pt", dev)
    scale = float(ck.get("scale", np.percentile(g[g > 0], 99.5)))
    gn = (g / scale).astype(np.float32)

    X_A = infer_fix_axis(model, gn, dev, fix_axis=2) * scale  # R-A 切片
    X_B = infer_fix_axis(model, gn, dev, fix_axis=1) * scale  # R-S 切片
    X_mean = 0.5 * (X_A + X_B)

    E_A = np.abs(X_A - hr); E_B = np.abs(X_B - hr)
    E_mean = np.abs(X_mean - hr); E_sinc = np.abs(g - hr)
    U = np.abs(X_A - X_B)
    U_s = ndimage.gaussian_filter(U, sigma=1.0)
    H = np.maximum(0.0, E_mean - E_sinc)

    # 控制量（均可从 LR 获得）
    g_grad = ndimage.generic_gradient_magnitude(g, ndimage.sobel)
    local_var = ndimage.uniform_filter(g ** 2, size=5) - ndimage.uniform_filter(g, size=5) ** 2
    resid_amp = np.abs(0.5 * ((X_A - g) + (X_B - g)))

    m = mask
    r_u_e, p_u_e = spearmanr(U_s[m], E_mean[m])
    r_u_h, p_u_h = spearmanr(U_s[m], H[m])
    r_u_ea, _ = spearmanr(U_s[m], E_A[m])
    r_u_eb, _ = spearmanr(U_s[m], E_B[m])
    r_u_g, _ = spearmanr(U_s[m], g_grad[m])
    r_u_v, _ = spearmanr(U_s[m], local_var[m])
    r_u_r, _ = spearmanr(U_s[m], resid_amp[m])
    # 偏相关：控制 Sinc 梯度、局部方差、残差幅值
    pr_grad = rank_partial_spearman(U_s, E_mean, g_grad, m)
    pr_var = rank_partial_spearman(U_s, E_mean, local_var, m)
    pr_resid = rank_partial_spearman(U_s, E_mean, resid_amp, m)

    # 分位数分析（U 的 10 个分位，误差均值）
    vals = U_s[m]; errs = E_mean[m]
    qs = np.quantile(vals, np.linspace(0, 1, 11))
    means = []
    for i in range(10):
        sel = (vals >= qs[i]) & (vals <= qs[i + 1])
        means.append(float(errs[sel].mean()) if sel.any() else float("nan"))

    harmful_ratio = float((H[m] > 0).mean())
    # 有害修正的幅度（相对于 Sinc 误差）
    harmful_mag = float(H[m].mean() / (E_sinc[m].mean() + 1e-9))

    return {
        "subject": s,
        "mask_voxels": int(m.sum()),
        "spearman_U_Errm": [float(r_u_e), float(p_u_e)],
        "spearman_U_Harmful": [float(r_u_h), float(p_u_h)],
        "spearman_U_EA": float(r_u_ea), "spearman_U_EB": float(r_u_eb),
        "spearman_U_grad": float(r_u_g), "spearman_U_var": float(r_u_v),
        "spearman_U_resid": float(r_u_r),
        "partial_U_Errm_grad": pr_grad,
        "partial_U_Errm_var": pr_var,
        "partial_U_Errm_resid": pr_resid,
        "error_by_U_decile": means,
        "harmful_voxel_ratio": harmful_ratio,
        "harmful_mag_ratio": harmful_mag,
        "maps": {"U": U_s, "E_mean": E_mean, "H": H, "E_sinc": E_sinc, "g": g, "hr": hr,
                 "X_A": X_A, "X_B": X_B, "X_mean": X_mean},
    }


def save_figure(rec, out_dir: Path):
    s = rec["subject"]; maps = rec["maps"]
    hr = maps["hr"]
    # 统一显示窗口
    fg = hr[hr > 0]; lo, hi = np.percentile(fg, [0.5, 99.5])
    win = lambda x: np.clip((x - lo) / (hi - lo), 0, 1)
    # 选一个病灶/结构丰富的 Axial 切片（固定 S=中心）
    cut = 2; center = hr.shape[cut] // 2
    sl = lambda a: np.take(a, center, axis=cut).T

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    items = [("HR", win(hr)), ("Sinc", win(maps["g"])), ("X_A (R-A)", win(maps["X_A"])),
             ("X_B (R-S)", win(maps["X_B"])), ("Disagreement U", maps["U"]),
             ("|X_mean - HR|", maps["E_mean"]), ("Harmful H", maps["H"]),
             ("|Sinc - HR|", maps["E_sinc"])]
    for ax, (title, arr) in zip(axes.ravel(), items):
        cmap = "gray" if "U" not in title and "Harm" not in title and "|" not in title else "hot"
        vmax = 1.0 if cmap == "gray" else None
        ax.imshow(sl(arr), cmap=cmap, vmin=0, vmax=vmax, origin="lower", aspect="equal")
        ax.set_title(title, fontsize=9); ax.axis("off")
    fig.suptitle(f"{s} — cross-plane disagreement vs reconstruction error (axial slice)", fontsize=11)
    fig.tight_layout()
    out = out_dir / f"{s}_disagreement_maps.png"
    fig.savefig(out, dpi=140, facecolor="white"); plt.close(fig)

    # U vs error（下采样散点）+ 分位数曲线
    m = rec["maps"]["U"] > 0  # placeholder
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    U = rec["maps"]["U"]; E = rec["maps"]["E_mean"]
    rng = np.random.default_rng(0)
    idx = rng.choice(U.size, size=min(200000, U.size), replace=False)
    axes[0].scatter(U.ravel()[idx], E.ravel()[idx], s=1, alpha=0.15, c="tab:blue")
    axes[0].set_xlabel("Cross-plane disagreement U"); axes[0].set_ylabel("|X_mean - HR|")
    axes[0].set_title(f"U vs error (Spearman r={rec['spearman_U_Errm'][0]:.3f})")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(range(1, 11), rec["error_by_U_decile"], marker="o")
    axes[1].set_xlabel("Disagreement decile (1=low, 10=high)")
    axes[1].set_ylabel("Mean |X_mean - HR|")
    axes[1].set_title("Error by disagreement decile")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out2 = out_dir / f"{s}_disagreement_stats.png"
    fig.savefig(out2, dpi=140, facecolor="white"); plt.close(fig)
    return out, out2


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    summaries = []
    for s in SUBJECTS:
        print(f"分析 {s} ...")
        rec = analyze_subject(s, dev)
        figs = save_figure(rec, FIG)
        summary = {k: v for k, v in rec.items() if k != "maps"}
        summary["figures"] = [str(f.relative_to(PROJECT_ROOT)) for f in figs]
        summaries.append(summary)
        print(f"  Spearman(U, E_mean)={summary['spearman_U_Errm'][0]:.3f} "
              f"Spearman(U, H)={summary['spearman_U_Harmful'][0]:.3f} "
              f"partial(grad)={summary['partial_U_Errm_grad']:.3f} "
              f"harmful_ratio={summary['harmful_voxel_ratio']:.3f}")

    (OUT / "cross_plane_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n汇总已保存: {OUT / 'cross_plane_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
