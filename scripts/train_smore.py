"""Day 3/4：简化 SMORE(3D) 自监督训练（支持 L1 / L1+SobelGradient）。

记录系统（Day 4 新增）：
- TensorBoard：results/tensorboard/<exp_name>/
- CSV：results/logs/csv/<exp_name>_train.csv / _val.csv
- 分项记录 train/val 的 total / L1 / gradient loss、learning rate、global step、epoch。

实验独立命名：lambda_grad=0 -> "day4_l1"；否则 -> "day4_grad_<lambda*100>"，
各自使用独立 checkpoint / tensorboard / csv 目录，不互相覆盖，也不覆盖 Day 3。

用法::

    python scripts/train_smore.py --loss l1                    # baseline
    python scripts/train_smore.py --loss l1_grad --lambda-grad 0.05
    python scripts/train_smore.py --overfit-only               # 阶段 A 过拟合测试
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_mri
from src.degradation import lowpass_along_axis
from src.edsr import EDSR, count_parameters
from src.losses import SobelGradientLoss, combined_loss
from src.self_supervised_dataset import SMOREDataset, _foreground_scale


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve(cfg: dict, key: str, default=None):
    for part in key.split("."):
        if not isinstance(cfg, dict) or part not in cfg:
            return default
        cfg = cfg[part]
    return cfg


def build_model(cfg: dict, device: torch.device) -> EDSR:
    m = resolve(cfg, "model", {})
    return EDSR(
        n_resblocks=m.get("n_resblocks", 8),
        n_feats=m.get("n_feats", 64),
        res_scale=m.get("res_scale", 0.1),
        n_channels=m.get("n_channels", 1),
    ).to(device)


def experiment_name(loss: str, lambda_grad: float) -> str:
    if loss == "l1" or lambda_grad == 0.0:
        return "day4_l1"
    return f"day4_grad_{int(round(lambda_grad * 100)):03d}"


# ---------------------------------------------------------------------------
# 阶段 A：最小过拟合测试（可选）
# ---------------------------------------------------------------------------
def stage_a_overfit(cfg: dict, device: torch.device) -> bool:
    print("\n===== 阶段 A：最小过拟合测试 =====")
    oc = cfg.get("overfit", {})
    n_patches = int(oc.get("n_patches", 8))
    steps = int(oc.get("steps", 400))
    lr = float(oc.get("lr", 2e-4))

    data = resolve(cfg, "data", {})
    g_path = PROJECT_ROOT / data["g_path"]
    r = float(data.get("r", 2.0))
    filter_name = data.get("filter_name", "gaussian")
    patch_size = int(data.get("patch_size", 32))

    vol = load_mri(g_path, canonical=True)
    g = vol.data.astype(np.float64)
    scale = _foreground_scale(g)
    g_norm = (g / scale).astype(np.float32)

    rng = np.random.default_rng(0)
    inputs, targets = [], []
    n_r = g.shape[0]
    for _ in range(n_patches):
        i = int(rng.integers(n_r // 4, n_r - n_r // 4))
        slc = g_norm[i, :, :]
        blurred = lowpass_along_axis(slc, axis=0, r=r, filter_name=filter_name)
        h, w = slc.shape
        i0 = int(rng.integers(0, h - patch_size)); j0 = int(rng.integers(0, w - patch_size))
        inputs.append(blurred[i0:i0 + patch_size, j0:j0 + patch_size])
        targets.append(slc[i0:i0 + patch_size, j0:j0 + patch_size])

    x = torch.from_numpy(np.stack(inputs)[:, None]).float().to(device)
    y = torch.from_numpy(np.stack(targets)[:, None]).float().to(device)

    model = build_model(cfg, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sobel = SobelGradientLoss().to(device)
    params_before = [p.detach().clone() for p in model.parameters()]
    loss_hist = []; grads_finite = True
    for step in range(1, steps + 1):
        opt.zero_grad()
        out = model(x)
        total, l1, grad = combined_loss(out, y, sobel, lambda_grad=0.0)
        total.backward()
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                grads_finite = False
        opt.step()
        loss_hist.append(float(l1.item()))
    params_changed = any(not torch.allclose(a, b) for a, b in zip(model.parameters(), params_before))
    final = float(F.l1_loss(model(x), y))
    ok = final < loss_hist[0] and params_changed and grads_finite
    print(f"  L1 {loss_hist[0]:.6f} -> {final:.6f}, 参数更新={params_changed}, 梯度有限={grads_finite}")
    print(f"  阶段 A 结果: {'通过' if ok else '失败'}")
    return ok


# ---------------------------------------------------------------------------
# 阶段 B：完整自监督训练（带完整记录）
# ---------------------------------------------------------------------------
def stage_b_train(cfg: dict, args, device: torch.device) -> dict:
    data = resolve(cfg, "data", {})
    tr = resolve(cfg, "training", {})
    g_path = PROJECT_ROOT / data["g_path"]
    seed = int(args.seed if args.seed is not None else cfg.get("experiment", {}).get("seed", 42))
    set_seed(seed)

    loss_name = args.loss
    lambda_grad = float(args.lambda_grad) if loss_name == "l1_grad" else 0.0
    exp = args.experiment_name or experiment_name(loss_name, lambda_grad)
    max_steps = int(args.max_steps if args.max_steps is not None else tr.get("max_steps", 3000))

    r = float(data.get("r", 2.0))
    # 支持多退化训练：--filters 提供列表则随机采样多个滤波器
    filter_name = args.filters if args.filters else data.get("filter_name", "gaussian")
    patch_size = int(data.get("patch_size", 32))
    patches_per_slice = int(data.get("patches_per_slice", 32))
    val_interval = int(data.get("val_interval", 5))
    batch_size = int(tr.get("batch_size", 16))
    lr = float(tr.get("lr", 1e-4))
    weight_decay = float(tr.get("weight_decay", 0.0))
    val_every = int(tr.get("val_every", 200))
    log_interval = int(tr.get("log_interval", 10))
    num_workers = int(tr.get("num_workers", 0))

    print(f"\n===== 阶段 B：{exp} (loss={loss_name}, lambda_grad={lambda_grad}) =====")

    # 共享 scale
    vol = load_mri(g_path, canonical=True)
    scale = _foreground_scale(vol.data.astype(np.float64))

    train_ds = SMOREDataset(g_path, r=r, filter_name=filter_name, patch_size=patch_size,
                            patches_per_slice=patches_per_slice, mode="train",
                            val_interval=val_interval, seed=seed, scale=scale,
                            noise_sigma=args.noise_sigma)
    val_ds = SMOREDataset(g_path, r=r, filter_name=filter_name, patch_size=patch_size,
                          patches_per_slice=patches_per_slice, mode="val",
                          val_interval=val_interval, seed=seed, scale=scale,
                          noise_sigma=args.noise_sigma)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        drop_last=True, pin_memory=(device.type == "cuda"))
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=(device.type == "cuda"))

    steps_per_epoch = max(len(train_loader), 1)
    print(f"训练切片={len(train_ds.r_indices)}, 验证切片={len(val_ds.r_indices)}, "
          f"steps/epoch={steps_per_epoch}, scale={scale:.4f}")

    model = build_model(cfg, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sobel = SobelGradientLoss().to(device)

    # 目录
    ckpt_dir = PROJECT_ROOT / "results" / "checkpoints" / exp
    tb_dir = PROJECT_ROOT / "results" / "tensorboard" / exp
    csv_dir = PROJECT_ROOT / "results" / "logs" / "csv"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    train_csv = csv_dir / f"{exp}_train.csv"
    val_csv = csv_dir / f"{exp}_val.csv"
    train_f = train_csv.open("w", newline="", encoding="utf-8")
    val_f = val_csv.open("w", newline="", encoding="utf-8")
    train_w = csv.writer(train_f)
    val_w = csv.writer(val_f)
    train_w.writerow(["global_step", "epoch", "train_total_loss", "train_l1_loss",
                      "train_gradient_loss", "learning_rate"])
    val_w.writerow(["global_step", "epoch", "val_total_loss", "val_l1_loss",
                    "val_gradient_loss", "learning_rate"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best_val = float("inf"); best_step = 0
    train_iter = iter(train_loader)
    t0 = time.time()

    def current_lr() -> float:
        return opt.param_groups[0]["lr"]

    for step in range(1, max_steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        out = model(x)
        total, l1, grad = combined_loss(out, y, sobel, lambda_grad)
        total.backward()
        opt.step()

        if step % log_interval == 0:
            epoch = step / steps_per_epoch
            train_w.writerow([step, round(epoch, 4), float(total.item()), float(l1.item()),
                              float(grad.item()), current_lr()])
            train_f.flush()
            writer.add_scalar("train/total_loss", total.item(), step)
            writer.add_scalar("train/l1_loss", l1.item(), step)
            writer.add_scalar("train/gradient_loss", grad.item(), step)
            writer.add_scalar("lr", current_lr(), step)

        if step % val_every == 0 or step == max_steps:
            model.eval()
            vl1 = vgrad = 0.0; vn = 0
            with torch.no_grad():
                for xv, yv in val_loader:
                    xv, yv = xv.to(device), yv.to(device)
                    ov = model(xv)
                    vl1 += float(F.l1_loss(ov, yv).item()) * xv.size(0)
                    vgrad += float(sobel(ov, yv).item()) * xv.size(0)
                    vn += xv.size(0)
            vl1 /= vn; vgrad /= vn
            vtotal = vl1 + lambda_grad * vgrad
            model.train()

            epoch = step / steps_per_epoch
            val_w.writerow([step, round(epoch, 4), round(vtotal, 8), round(vl1, 8),
                            round(vgrad, 8), current_lr()])
            val_f.flush()
            writer.add_scalar("val/total_loss", vtotal, step)
            writer.add_scalar("val/l1_loss", vl1, step)
            writer.add_scalar("val/gradient_loss", vgrad, step)
            print(f"  step {step:5d}/{max_steps}  train L1={l1.item():.6f} grad={grad.item():.6f} | "
                  f"val L1={vl1:.6f} grad={vgrad:.6f}")

            if vtotal < best_val:
                best_val = vtotal; best_step = step
                torch.save({"step": step, "model_state_dict": model.state_dict(),
                            "val_l1": vl1, "val_grad": vgrad, "val_total": vtotal,
                            "scale": scale, "lambda_grad": lambda_grad, "loss": loss_name,
                            "config": cfg},
                           ckpt_dir / "best.pt")

    elapsed = time.time() - t0
    peak_mem = (torch.cuda.max_memory_allocated() / 1024**2) if device.type == "cuda" else 0.0

    torch.save({"step": max_steps, "model_state_dict": model.state_dict(),
                "scale": scale, "lambda_grad": lambda_grad, "loss": loss_name, "config": cfg},
               ckpt_dir / "last.pt")

    writer.close(); train_f.close(); val_f.close()

    summary = {
        "experiment": exp, "loss": loss_name, "lambda_grad": lambda_grad, "seed": seed,
        "scale": scale, "max_steps": max_steps, "best_val_total": best_val,
        "best_step": best_step, "train_seconds": round(elapsed, 2),
        "peak_gpu_mb": round(peak_mem, 2), "n_params": count_parameters(model),
        "n_train_slices": len(train_ds.r_indices), "n_val_slices": len(val_ds.r_indices),
        "checkpoint_dir": str(ckpt_dir.relative_to(PROJECT_ROOT)),
        "tensorboard_dir": str(tb_dir.relative_to(PROJECT_ROOT)),
        "train_csv": str(train_csv.relative_to(PROJECT_ROOT)),
        "val_csv": str(val_csv.relative_to(PROJECT_ROOT)),
    }
    (PROJECT_ROOT / "results" / "logs" / csv_dir.name / f"{exp}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"最佳验证 total={best_val:.6f} @ step {best_step}；耗时 {elapsed:.1f}s，峰值显存 {peak_mem:.1f} MB")
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Day 3/4 SMORE 自监督训练")
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "smore_baseline.yaml")
    p.add_argument("--overfit-only", action="store_true", help="只跑阶段 A 过拟合测试")
    p.add_argument("--loss", default="l1", choices=["l1", "l1_grad"])
    p.add_argument("--lambda-grad", type=float, default=0.0)
    p.add_argument("--experiment-name", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--g-path", default=None, help="覆盖 config 里的 data.g_path（多受试者）")
    p.add_argument("--filters", nargs="*", default=None, help="多退化训练：滤波器列表，如 --filters gaussian box fermi")
    p.add_argument("--noise-sigma", type=float, default=0.0, help="训练输入加 Rician 噪声（归一化强度单位）")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.g_path:
        cfg["data"]["g_path"] = args.g_path
    device = get_device()
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}, PyTorch {torch.__version__}")

    if args.overfit_only:
        return 0 if stage_a_overfit(cfg, device) else 1

    summary = stage_b_train(cfg, args, device)
    print(f"\n完成。summary: {json.dumps(summary, ensure_ascii=False, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
