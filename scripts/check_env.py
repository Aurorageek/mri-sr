"""检查 Python / PyTorch / CUDA / GPU 与关键依赖版本，并保存 JSON 报告。

用法::

    python scripts/check_env.py
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import matplotlib
import nibabel
import numpy as np
import torch


def collect() -> dict:
    report = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "conda_env": sys.prefix,
        },
        "numpy": np.__version__,
        "nibabel": nibabel.__version__,
        "matplotlib": matplotlib.__version__,
        "torch": {
            "version": torch.__version__,
            "cuda_built": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "device_count": torch.cuda.device_count(),
        },
    }
    if torch.cuda.is_available():
        report["torch"].update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "device_capability": list(torch.cuda.get_device_capability(0)),
                "total_memory_gb": round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
                ),
            }
        )
        # 做一次真实的 GPU 张量运算，确认驱动/CUDA 栈可执行
        x = torch.randn(64, 64, device="cuda")
        y = x @ x.T
        torch.cuda.synchronize()
        report["torch"]["gpu_smoke_test"] = {
            "operation": "64x64 matmul on cuda",
            "ok": bool(torch.isfinite(y).all().item()),
            "sample_value": float(y[0, 0].item()),
        }
    return report


def main() -> int:
    report = collect()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    out = PROJECT_ROOT / "results" / "env_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n环境报告已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
