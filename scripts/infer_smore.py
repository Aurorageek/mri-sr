"""Day 3：简化 SMORE(3D) 三维推理。

流程（对应论文 Algorithm 1 的简化推理）：
1. 读取 g（Sinc 插值结果）；
2. 提取包含低分辨率 R 方向的 R-A 切片（固定 S）；
3. 用训练好的 EDSR 沿 R（切片 axis 0）去模糊；
4. 重组为三维体数据；
5. 保存 NIfTI（affine 与 g 一致，网格与 Day 2 评价网格一致）。

训练在 A-S 平面（模糊沿 A = 切片 axis 0）完成，推理在 R-A 平面（待去模糊轴
R = 切片 axis 0）执行——两者的模糊方向都是 patch 的第 0 个空间轴，因此可迁移。

用法::

    python scripts/infer_smore.py
    python scripts/infer_smore.py --checkpoint results/checkpoints/smore_baseline/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nibabel as nib

from src.dataset import load_mri
from src.edsr import EDSR


def load_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[EDSR, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    m = cfg.get("model", {})
    model = EDSR(
        n_resblocks=m.get("n_resblocks", 8),
        n_feats=m.get("n_feats", 64),
        res_scale=m.get("res_scale", 0.1),
        n_channels=m.get("n_channels", 1),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def infer_volume(model: EDSR, g: np.ndarray, device: torch.device, batch_size: int = 4) -> np.ndarray:
    """对 g 逐 R-A 切片（固定 S）沿 R 去模糊，返回去模糊后的体数据（g 同形）。

    g 已归一化（除以 scale），输出仍是归一化域，由调用方乘回 scale。
    """
    # g shape = (R, A, S)
    n_r, n_a, n_s = g.shape
    out = np.zeros_like(g, dtype=np.float32)
    with torch.no_grad():
        for s0 in range(0, n_s, batch_size):
            s1 = min(s0 + batch_size, n_s)
            # 每个 s 一张 (R, A) 切片；堆叠成 [B,1,R,A]
            batch = []
            for s in range(s0, s1):
                slc = g[:, :, s]  # (R, A)
                batch.append(slc[None, None, ...])  # [1,1,R,A]
            x = torch.from_numpy(np.concatenate(batch, axis=0)).float().to(device)
            y = model(x).cpu().numpy()  # [B,1,R,A]
            for k, s in enumerate(range(s0, s1)):
                out[:, :, s] = y[k, 0]
    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Day 3 SMORE 三维推理")
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "smore_baseline.yaml")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="checkpoint 路径；默认用 config 里的 best.pt")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--output", type=Path, default=None,
                   help="输出 NIfTI 路径；默认 <g 目录>/<subject>_smore_r2.nii.gz")
    p.add_argument("--g-path", default=None, help="覆盖 config 里的 data.g_path")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.g_path:
        cfg["data"]["g_path"] = args.g_path
    data = cfg["data"]
    subject = cfg["experiment"]["subject"]
    g_path = PROJECT_ROOT / data["g_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    ckpt_path = args.checkpoint or (PROJECT_ROOT / cfg["training"]["checkpoint_dir"] / "best.pt")
    if not ckpt_path.exists():
        print(f"checkpoint 不存在: {ckpt_path}")
        return 1
    model, ckpt = load_checkpoint(ckpt_path, device)
    scale = float(ckpt.get("scale", 1.0))
    print(f"加载 checkpoint: {ckpt_path}")
    print(f"checkpoint step={ckpt.get('step')}, val_l1={ckpt.get('val_l1')}, scale={scale:.4f}")

    vol = load_mri(g_path, canonical=True)
    g = vol.data.astype(np.float64)
    g_norm = (g / scale).astype(np.float32)
    print(f"g shape={g.shape}, 推理方向=R（axis 0），切面=R-A（固定 S）")

    out_norm = infer_volume(model, g_norm, device, batch_size=args.batch_size)
    out = out_norm * scale  # 回到原始强度域

    out_path = args.output or (g_path.parent / f"{subject}_smore_r2.nii.gz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out.astype(np.float32), vol.affine), str(out_path))

    print(f"推理结果已保存: {out_path}")
    print(f"  输出 shape={out.shape}, affine 与 g 一致, orientation={vol.orientation}")

    # 记录推理元数据
    meta = {
        "subject": subject,
        "input": str(g_path.relative_to(PROJECT_ROOT)),
        "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)),
        "scale": scale,
        "output": str(out_path.relative_to(PROJECT_ROOT)),
        "shape": list(out.shape),
        "inference_planes": "R-A (fix S), deblur along axis 0 = R",
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
