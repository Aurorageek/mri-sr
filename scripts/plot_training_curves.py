"""Day 4：从 CSV 真实日志绘制训练曲线。

生成 4 张图（保存到 results/figures/training_curves/）：
1. Training Total Loss vs step
2. Validation Total Loss vs step
3. L1 Loss vs step（训练与验证）
4. Gradient Loss vs step（训练与验证）

注意：不同 lambda 的 Total Loss 定义不同，不能直接比绝对大小；跨实验应优先看
L1 / Gradient 分项曲线。图中同时标注 lambda 与实验名，只读真实 CSV，不伪造数据。

用法::

    python scripts/plot_training_curves.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = PROJECT_ROOT / "results" / "logs" / "csv"
OUT_DIR = PROJECT_ROOT / "results" / "figures" / "training_curves"


def lambda_from_name(name: str) -> float:
    if name.endswith("_l1"):
        return 0.0
    # day4_grad_010 -> 0.10
    return int(name.split("_")[-1]) / 100.0


def load_csv(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        data = {c: [] for c in cols}
        for row in reader:
            for c in cols:
                data[c].append(float(row[c]))
    return data


def discover() -> list[tuple[str, float, dict, dict]]:
    """返回 [(exp, lambda, train_data, val_data), ...]，按 lambda 升序。"""
    exps = []
    for train_csv in sorted(CSV_DIR.glob("*_train.csv")):
        exp = train_csv.name[: -len("_train.csv")]
        val_csv = CSV_DIR / f"{exp}_val.csv"
        if not val_csv.exists():
            continue
        exps.append((exp, lambda_from_name(exp), load_csv(train_csv), load_csv(val_csv)))
    exps.sort(key=lambda t: t[1])
    return exps


def _smooth(x: np.ndarray, w: int = 5) -> np.ndarray:
    if len(x) < w:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def _plot_pair(ax, train_d, val_d, xkey, train_ykey, val_ykey, label):
    xt = np.array(train_d[xkey]); yt = np.array(train_d[train_ykey])
    ax.plot(xt, yt, lw=0.7, alpha=0.35)
    ax.plot(xt, _smooth(yt), lw=1.6, label=f"{label} train")
    if val_d:
        xv = np.array(val_d[xkey]); yv = np.array(val_d[val_ykey])
        ax.plot(xv, yv, lw=1.6, marker="o", ms=3, label=f"{label} val")


def main() -> int:
    exps = discover()
    if not exps:
        print(f"在 {CSV_DIR} 未找到 *_train.csv / *_val.csv")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 图一/图二：Total Loss（train / val 分开）
    for split, ykey, title, fname in [
        ("train", "train_total_loss", "Training Total Loss vs Global Step", "fig1_train_total.png"),
        ("val", "val_total_loss", "Validation Total Loss vs Global Step", "fig2_val_total.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for exp, lam, td, vd in exps:
            d = td if split == "train" else vd
            if not d:
                continue
            x = np.array(d["global_step"]); y = np.array(d[ykey])
            ax.plot(x, y, lw=1.6, label=f"{exp} (λ={lam:g})")
        ax.set_xlabel("Global Step")
        ax.set_ylabel(ykey)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT_DIR / fname, dpi=140); plt.close(fig)

    # 图三/图四：L1 与 Gradient（train+val 同图，强调可比的 L1/Grad）
    for ykey, title, fname in [
        ("l1_loss", "L1 Loss vs Global Step", "fig3_l1.png"),
        ("gradient_loss", "Gradient Loss vs Global Step", "fig4_gradient.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for exp, lam, td, vd in exps:
            tk = "train_" + ykey; vk = "val_" + ykey
            _plot_pair(ax, td, vd, "global_step", tk, vk, label=f"{exp} (λ={lam:g})")
        ax.set_xlabel("Global Step")
        ax.set_ylabel(ykey)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT_DIR / fname, dpi=140); plt.close(fig)

    print(f"训练曲线已保存到 {OUT_DIR}")
    for f in sorted(OUT_DIR.glob("*.png")):
        print("  ", f.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
