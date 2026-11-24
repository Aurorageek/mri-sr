"""为指定滤波器生成实际 LR（Protocol B 用）。

生成 g_{filter} = sinc_interpolate(downsample(lowpass_{filter}(HR)))，
保存为 NIfTI（与 HR 同网格同 affine）。

用法::

    python improvements/scripts/generate_lr.py --subject IXI002-Guys-0828-T1 --filter box
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_mri
from src.degradation import degrade_through_plane, sinc_interpolate


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="生成指定滤波器的实际 LR")
    p.add_argument("--subject", default="IXI002-Guys-0828-T1")
    p.add_argument("--filter", default="box", choices=["gaussian", "box", "fermi"])
    p.add_argument("--r", type=float, default=2.0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    proc = PROJECT_ROOT / "data" / "processed" / args.subject
    hr_vol = load_mri(proc / f"{args.subject}_HR.nii.gz", canonical=True)
    hr = hr_vol.data.astype(np.float64)

    deg = degrade_through_plane(hr, axis=0, r=args.r, filter_name=args.filter)
    g = sinc_interpolate(deg.lr, axis=0, r=args.r, target_n=hr.shape[0])

    out = proc / f"{args.subject}_sinc_r{int(args.r)}_{args.filter}.nii.gz"
    nib.save(nib.Nifti1Image(g.astype(np.float32), hr_vol.affine), str(out))
    print(f"已保存 {out}  shape={g.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
