"""Day 5：多受试者正式评价与汇总。

对每个受试者，用统一协议计算 Sinc / SMORE-L1 / SMORE-Grad 的 PSNR/SSIM/MAE/
边缘误差，输出 per_subject.csv、summary.csv、ablation.csv 和 JSON；对相同
受试者做配对比较，样本量足够时执行 paired Wilcoxon signed-rank test。

方法文件命名约定（data/processed/<subject>/）：
- Sinc        : <subject>_sinc_r2.nii.gz
- SMORE-L1    : <subject>_smore_l1.nii.gz
- SMORE-Grad  : <subject>_smore_grad.nii.gz

用法::

    python scripts/eval_multisubject.py --subjects IXI013-HH-1212-T1 IXI014-HH-1236-T1 ...
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_mri
from src.metrics import brain_mask, evaluate_with_edge

METHODS = [
    ("sinc", "Sinc", "{s}_sinc_r2.nii.gz"),
    ("smore_l1", "SMORE-L1", "{s}_smore_l1.nii.gz"),
    ("smore_grad", "SMORE-Grad", "{s}_smore_grad.nii.gz"),
]

METRIC_KEYS = ["psnr_db", "ssim", "mae", "gradient_mae",
               "through_plane_gradient_mae", "gradient_3d_mae"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Day 5 多受试者评价")
    p.add_argument("--subjects", nargs="*", default=None, help="受试者 ID 列表（默认自动发现）")
    p.add_argument("--r", type=float, default=2.0)
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "metrics")
    return p.parse_args(argv)


def discover_subjects() -> list[str]:
    proc = PROJECT_ROOT / "data" / "processed"
    return sorted(d.name for d in proc.iterdir() if d.is_dir())


def load_method(subject: str, key: str) -> np.ndarray | None:
    for mkey, _, pat in METHODS:
        if mkey == key:
            p = PROJECT_ROOT / "data" / "processed" / subject / pat.format(s=subject)
            if not p.exists():
                return None
            return load_mri(p, canonical=True).data.astype(np.float64)
    return None


def wilcoxon_p(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    d = np.array(a) - np.array(b)
    if np.all(d == 0):
        return 1.0
    try:
        return float(wilcoxon(a, b).pvalue)
    except Exception:  # noqa: BLE001
        return None


def main(argv=None) -> int:
    args = parse_args(argv)
    subjects = args.subjects or discover_subjects()
    subjects = [s for s in subjects if (PROJECT_ROOT / "data" / "processed" / s / f"{s}_HR.nii.gz").exists()]

    rows: list[dict] = []
    per_subject_metrics: dict[str, dict] = {}

    for s in subjects:
        hr_path = PROJECT_ROOT / "data" / "processed" / s / f"{s}_HR.nii.gz"
        hr_vol = load_mri(hr_path, canonical=True)
        hr = hr_vol.data.astype(np.float64)
        spacing = tuple(float(v) for v in hr_vol.voxel_spacing)
        mask = brain_mask(hr)
        per_subject_metrics[s] = {}
        for mkey, label, _ in METHODS:
            vol = load_method(s, mkey)
            if vol is None:
                continue
            m = evaluate_with_edge(hr, vol, mask, spacing=spacing, through_plane_axis=0)
            per_subject_metrics[s][mkey] = m
            rows.append({
                "subject_id": s, "degradation_ratio": args.r, "method": mkey,
                "lambda": {"sinc": None, "smore_l1": 0.0, "smore_grad": None}[mkey],
                "psnr": round(m["psnr_db"], 4), "ssim": round(m["ssim"], 5),
                "mae": round(m["mae"], 4),
                "gradient_error": round(m["gradient_mae"], 4),              # 历史遗留 S 轴
                "through_plane_gradient_error": round(m["through_plane_gradient_mae"], 4),
                "gradient_3d_error": round(m["gradient_3d_mae"], 4),
            })

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # per_subject.csv
    per_csv = out_dir / "per_subject.csv"
    with per_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subject_id", "degradation_ratio", "method",
                                          "lambda", "psnr", "ssim", "mae", "gradient_error",
                                          "through_plane_gradient_error", "gradient_3d_error"])
        w.writeheader(); w.writerows(rows)

    # summary.csv：每方法每指标的 mean / std
    row_keys = {
        "psnr_db": "psnr", "ssim": "ssim", "mae": "mae",
        "gradient_mae": "gradient_error",
        "through_plane_gradient_mae": "through_plane_gradient_error",
        "gradient_3d_mae": "gradient_3d_error",
    }
    summary_rows = []
    for mkey, label, _ in METHODS:
        rec = {"method": mkey, "label": label, "n_subjects": 0}
        for mk in METRIC_KEYS:
            vals = [r[row_keys[mk]] for r in rows if r["method"] == mkey]
            rec[f"{mk}_mean"] = round(float(np.mean(vals)), 4) if vals else None
            rec[f"{mk}_std"] = round(float(np.std(vals)), 4) if vals else None
            rec["n_subjects"] = len(vals)
        summary_rows.append(rec)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        cols = list(summary_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(summary_rows)

    # ablation.csv：相对 Sinc / 相对 SMORE-L1 的配对差异
    abl_rows = []
    common = [s for s in subjects
              if all(k in per_subject_metrics.get(s, {}) for k in ("sinc", "smore_l1", "smore_grad"))]
    if common:
        for mk in METRIC_KEYS:
            sinc = [per_subject_metrics[s]["sinc"][mk] for s in common]
            l1 = [per_subject_metrics[s]["smore_l1"][mk] for s in common]
            grad = [per_subject_metrics[s]["smore_grad"][mk] for s in common]
            abl_rows.append({
                "metric": mk,
                "smore_l1_minus_sinc_mean": round(float(np.mean(np.array(l1) - np.array(sinc))), 4),
                "smore_grad_minus_smore_l1_mean": round(float(np.mean(np.array(grad) - np.array(l1))), 4),
                "wilcoxon_smore_l1_vs_sinc_p": wilcoxon_p(sinc, l1),
                "wilcoxon_smore_grad_vs_smore_l1_p": wilcoxon_p(l1, grad),
                "n_pairs": len(common),
            })
    with (out_dir / "ablation.csv").open("w", newline="", encoding="utf-8") as f:
        cols = list(abl_rows[0].keys()) if abl_rows else []
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(abl_rows)

    # JSON（含逐例）
    payload = {
        "subjects": subjects,
        "degradation_ratio": args.r,
        "per_subject": {s: {k: {kk: (vv if not isinstance(vv, float) or np.isfinite(vv) else None)
                                for kk, vv in v.items()} for k, v in per_subject_metrics.get(s, {}).items()}
                        for s in subjects},
        "summary": summary_rows,
        "ablation": abl_rows,
    }
    (out_dir / "day5_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"受试者数: {len(subjects)}，逐例行数: {len(rows)}")
    print("summary:")
    for r in summary_rows:
        print(f"  {r['label']:<12} PSNR {r.get('psnr_db_mean')} ± {r.get('psnr_db_std')}  "
              f"SSIM {r.get('ssim_mean')}  TP-Grad {r.get('through_plane_gradient_mae_mean')}  "
              f"3D-Grad {r.get('gradient_3d_mae_mean')}  (n={r['n_subjects']})")
    print("\nablation (配对差异):")
    for r in abl_rows:
        print(f"  {r['metric']:<28} L1-Sinc={r['smore_l1_minus_sinc_mean']:+}  "
              f"Grad-L1={r['smore_grad_minus_smore_l1_mean']:+}  "
              f"p(L1vsSinc)={r['wilcoxon_smore_l1_vs_sinc_p']}  p(GradvsL1)={r['wilcoxon_smore_grad_vs_smore_l1_p']}")
    print(f"\n输出: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
