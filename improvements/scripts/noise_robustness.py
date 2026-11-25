"""EXP019：噪声鲁棒性 pilot（备选路线 A）。

对论文对齐 LR 加 Rician 噪声（σ_eval），比较：
- Sinc（噪声输入本身）
- paper-aligned SMORE（干净训练）
- paper-aligned SMORE + Rician 噪声增强训练（σ=0.05）

用法::

    python improvements/scripts/noise_robustness.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_mri
from src.edsr import EDSR
from src.metrics import brain_mask, psnr, ssim

SUBJECTS = ["IXI002-Guys-0828-T1", "IXI012-HH-1211-T1"]
SIGMAS = [0.0, 0.05, 0.1, 0.2]
OUT = PROJECT_ROOT / "improvements" / "results" / "phase4"


def load_model(p, dev):
    ck = torch.load(p, map_location=dev); m = ck.get("config", {}).get("model", {})
    model = EDSR(m.get("n_resblocks", 8), m.get("n_feats", 64), m.get("res_scale", 0.1)).to(dev)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    return model, float(ck.get("scale", 1.0))


def infer(model, g, dev, bs=4):
    out = np.zeros_like(g, dtype=np.float32)
    n_s = g.shape[2]
    with torch.no_grad():
        for s0 in range(0, n_s, bs):
            s1 = min(s0 + bs, n_s)
            batch = np.concatenate([g[:, :, s][None, None] for s in range(s0, s1)], 0)
            y = model(torch.from_numpy(batch).float().to(dev)).cpu().numpy()
            for k, s in enumerate(range(s0, s1)):
                out[:, :, s] = y[k, 0]
    return out


def rician(x, sigma, rng):
    n1 = rng.standard_normal(x.shape) * sigma
    n2 = rng.standard_normal(x.shape) * sigma
    return np.sqrt((x + n1) ** 2 + n2 ** 2)


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for s in SUBJECTS:
        proc = PROJECT_ROOT / "data" / "processed" / s
        hr = load_mri(proc / f"{s}_HR.nii.gz", canonical=True).data.astype(np.float64)
        g = load_mri(proc / f"{s}_sinc_r2_paper.nii.gz", canonical=True).data.astype(np.float64)
        mask = brain_mask(hr)
        m_clean, sc_clean = load_model(PROJECT_ROOT / "results/checkpoints" / f"{s}_paperaligned" / "best.pt", dev)
        m_noise, sc_noise = load_model(PROJECT_ROOT / "results/checkpoints" / f"{s}_paperaligned_noise005" / "best.pt", dev)

        rows = []
        for sg in SIGMAS:
            rng = np.random.default_rng(1234)
            gn = rician(g / sc_clean, sg, rng)  # 归一化域加噪
            g_noisy = gn * sc_clean
            entry = {"sigma": sg,
                     "sinc_psnr": psnr(hr, g_noisy, mask),
                     "sinc_ssim": ssim(hr, g_noisy, mask, win_size=7)}
            for name, model, sc in [("clean", m_clean, sc_clean), ("noise-trained", m_noise, sc_noise)]:
                inp = (gn * (sc_clean / sc)).astype(np.float32) if sc != sc_clean else gn.astype(np.float32)
                rec = infer(model, inp, dev) * sc
                entry[f"{name}_psnr"] = psnr(hr, rec.astype(np.float64), mask)
                entry[f"{name}_ssim"] = ssim(hr, rec.astype(np.float64), mask, win_size=7)
            rows.append(entry)
        results[s] = rows
        print(f"== {s} ==")
        print(f"{'sigma':>6} {'Sinc':>8} {'clean':>8} {'noise-tr':>9} {'d(noise-clean)':>15}")
        for e in rows:
            print(f"{e['sigma']:>6.2f} {e['sinc_psnr']:>8.3f} {e['clean_psnr']:>8.3f} "
                  f"{e['noise-trained_psnr']:>9.3f} {e['noise-trained_psnr']-e['clean_psnr']:>+15.3f}", flush=True)
    (OUT / "noise_robustness_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", OUT / "noise_robustness_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
