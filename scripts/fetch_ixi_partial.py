"""从 IXI 官方服务器按需提取少量 T1 受试者（不下载整个 4.8 GB 数据集）。

背景
----
IXI 官方只提供按模态打包的完整 tar：
https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar
（约 4.84 GB，未压缩 tar，服务器支持 HTTP Range）。

本脚本利用 HTTP Range + tar 流式读取，只读取压缩包开头约几十 MB，
提取前 N 个 .nii.gz 受试者后立即关闭连接。不会下载完整数据集，
也不会使用任何第三方镜像。

用法::

    python scripts/fetch_ixi_partial.py --n 3
    python scripts/fetch_ixi_partial.py --n 3 --out-dir data/raw/ixi

注意：IXI 数据仅用于学术研究，使用时请遵守官方页面上的使用条件。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nibabel as nib

IXI_URL = "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar"
USER_AGENT = "MRI-SR-course-project/1.0 (academic use; partial range request)"


class CountingReader:
    """包装 HTTP 响应，统计实际下载的字节数。"""

    def __init__(self, raw):
        self._raw = raw
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self._raw.read(size)
        self.bytes_read += len(chunk)
        return chunk

    def close(self) -> None:
        self._raw.close()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _verify_nifti(path: Path) -> tuple:
    """确认文件是可读的三维 NIfTI（只读 header，不加载全部数据）。"""
    img = nib.load(str(path))
    shape = tuple(int(s) for s in img.shape)
    if len(shape) != 3 or min(shape) < 2:
        raise ValueError(f"{path.name} 不是有效三维 NIfTI，shape={shape}")
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return shape, zooms


def fetch_ixi_t1(n: int, out_dir: Path) -> list[dict]:
    """流式读取官方 tar，提取前 n 个 T1 受试者。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 已存在的文件先统计，避免重复下载
    existing = sorted(out_dir.glob("*.nii.gz"))
    existing_names = {p.name for p in existing}
    records: list[dict] = []
    for path in existing:
        shape, zooms = _verify_nifti(path)
        records.append(
            {
                "file": path.name,
                "local_path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "shape": list(shape),
                "voxel_spacing_mm": list(zooms),
                "source": "already present",
            }
        )
    if len(records) >= n:
        print(f"本地已有 {len(records)} 个文件（>= {n}），无需下载。")
        return records

    request = urllib.request.Request(
        IXI_URL,
        headers={
            "Range": "bytes=0-",  # 只读取流，提取足够样本后主动关闭
            "User-Agent": USER_AGENT,
            "Accept": "application/x-tar",
        },
    )

    print(f"正在从官方源流式读取: {IXI_URL}")
    print(f"目标: 提取 {n} 名受试者（不会保存完整 tar）")

    response = urllib.request.urlopen(request, timeout=120)
    reader = CountingReader(response)
    try:
        status = getattr(response, "status", None)
        print(f"HTTP 状态: {status} (206 表示 Range 生效)")

        with tarfile.open(fileobj=reader, mode="r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = Path(member.name).name  # 扁平化，防止路径穿越
                if not name.lower().endswith(".nii.gz"):
                    continue
                if name in existing_names:
                    continue  # 跳过已存在文件，避免重复计数/覆盖

                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                payload = extracted.read()
                dest = out_dir / name
                dest.write_bytes(payload)

                shape, zooms = _verify_nifti(dest)
                sha = _sha256(dest)
                records.append(
                    {
                        "file": name,
                        "local_path": str(dest.relative_to(PROJECT_ROOT)),
                        "bytes": dest.stat().st_size,
                        "sha256": sha,
                        "shape": list(shape),
                        "voxel_spacing_mm": list(zooms),
                        "source": IXI_URL,
                        "tar_member": member.name,
                    }
                )
                print(f"  [{len(records)}/{n}] {name}  shape={shape}  "
                      f"spacing={tuple(round(z, 3) for z in zooms)} mm")

                if len(records) >= n:
                    print("已达到目标数量，主动关闭连接，停止下载。")
                    break
    finally:
        try:
            reader.close()
        except Exception:
            pass

    return records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 IXI 官方 tar 中流式提取少量 T1 受试者")
    parser.add_argument("--n", type=int, default=3, help="需要的受试者数量（默认 3）")
    parser.add_argument(
        "--out-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "ixi",
        help="输出目录（默认 data/raw/ixi）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n < 1:
        print("--n 必须 >= 1")
        return 2

    started = datetime.now(timezone.utc)
    try:
        records = fetch_ixi_t1(args.n, args.out_dir)
    except (urllib.error.URLError, tarfile.TarError, OSError, ValueError) as exc:
        print(f"\n[下载失败] {type(exc).__name__}: {exc}")
        print("\n可以手动下载后放到以下目录：")
        print(f"  {args.out_dir.resolve()}")
        print("官方页面: https://brain-development.org/ixi-dataset/")
        print("官方文件: IXI-T1.tar（约 4.84 GB）；也可以只复制其中任意 2-3 个 .nii.gz。")
        return 1

    manifest = {
        "dataset": "IXI",
        "modality": "T1",
        "source_url": IXI_URL,
        "license_note": "IXI dataset is provided for academic research; see the official dataset page.",
        "fetched_at_utc": started.isoformat(),
        "requested_subjects": args.n,
        "files": records,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n成功获取 {len(records)} 个文件，清单已保存: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
