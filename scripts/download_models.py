#!/usr/bin/env python3
"""Download model files for a provider into ~/.local/share/kde-tts-daemon/models/.

Idempotent: skips files that already exist with the expected size (±5%).
Downloads to <name>.partial and atomically renames on completion.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import die, info, models_dir, warn  # noqa: E402  (sibling import)

# (name, url, approx_size_bytes)
KOKORO_FILES: list[tuple[str, str, int]] = [
    (
        "kokoro-v1.0.int8.onnx",
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx",
        88 * 1024 * 1024,
    ),
    (
        "voices-v1.0.bin",
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
        27 * 1024 * 1024,
    ),
]

KOKORO_FP16_GPU_FILES: list[tuple[str, str, int]] = [
    (
        "kokoro-v1.0.fp16-gpu.onnx",
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.fp16-gpu.onnx",
        169 * 1024 * 1024,
    ),
]


def _size_ok(path: Path, expected: int) -> bool:
    if not path.exists():
        return False
    actual = path.stat().st_size
    return abs(actual - expected) <= max(expected * 0.05, 1 * 1024 * 1024)


def _download(url: str, dest: Path, expected: int) -> None:
    partial = dest.with_suffix(dest.suffix + ".partial")
    info(f"fetching {dest.name} ({expected // (1024 * 1024)} MB est.)")
    last_pct = -1
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length") or expected)
        with partial.open("wb") as out:
            read = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                read += len(chunk)
                if total:
                    pct = int(read * 100 / total)
                    if pct != last_pct and pct % 5 == 0:
                        print(f"    {pct:>3}%  ({read // (1024 * 1024)}/{total // (1024 * 1024)} MB)", file=sys.stderr)
                        last_pct = pct
    partial.rename(dest)


def _fetch_set(files: list[tuple[str, str, int]], dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name, url, expected in files:
        dest = dest_dir / name
        if _size_ok(dest, expected):
            info(f"skipping {name} (already present, size ok)")
            continue
        if dest.exists():
            warn(f"{name} exists but size is off; redownloading")
            dest.unlink()
        try:
            _download(url, dest, expected)
        except Exception as e:
            die(f"failed to download {url}: {e}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="download_models.py")
    parser.add_argument("provider", nargs="?", default="kokoro-onnx")
    parser.add_argument(
        "--with-fp16-gpu",
        action="store_true",
        help="Also download GPU FP16 variant — the daemon uses it on CUDA/ROCm",
    )
    args = parser.parse_args(argv)

    dest = models_dir()
    if args.provider == "kokoro-onnx":
        _fetch_set(KOKORO_FILES, dest)
        if args.with_fp16_gpu:
            _fetch_set(KOKORO_FP16_GPU_FILES, dest)
    else:
        die(f"download_models: provider {args.provider!r} has no built-in download spec yet")
    info(f"models in {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
