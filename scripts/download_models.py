#!/usr/bin/env python3
"""Download model files for a provider into ~/.local/share/neural-tts-daemon/models/.

Idempotent: skips files that already exist with the expected size (±5%).
Downloads to <name>.partial and atomically renames on completion.

For providers backed by Hugging Face (e.g. longcat-audiodit) we shell out to
the provider's own venv python so huggingface_hub's snapshot_download handles
resumption, size verification, and atomic moves natively.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import die, info, models_dir, repo_root, warn  # noqa: E402  (sibling import)

# (name, url, approx_size_bytes)
# Default: FP16 model + voices. FP16 is faster than INT8 on both CPU (modern
# AVX-512) and GPU; only download INT8 explicitly via --with-int8 if you
# want the smaller (88 MB vs 169 MB) variant for constrained environments.
KOKORO_FILES: list[tuple[str, str, int]] = [
    (
        "kokoro-v1.0.fp16-gpu.onnx",
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.fp16-gpu.onnx",
        169 * 1024 * 1024,
    ),
    (
        "voices-v1.0.bin",
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
        27 * 1024 * 1024,
    ),
]

KOKORO_INT8_FILES: list[tuple[str, str, int]] = [
    (
        "kokoro-v1.0.int8.onnx",
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx",
        88 * 1024 * 1024,
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


LONGCAT_HF_REPO = "meituan-longcat/LongCat-AudioDiT-1B"
LONGCAT_DIR_NAME = "longcat-audiodit-1b"


def _hf_snapshot(venv_python: Path, repo_id: str, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    info(f"fetching {repo_id} into {target} (this can take a few minutes)")
    snippet = (
        "from huggingface_hub import snapshot_download;"
        f"snapshot_download(repo_id={repo_id!r}, local_dir={str(target)!r})"
    )
    result = subprocess.run([str(venv_python), "-c", snippet])
    if result.returncode != 0:
        die(f"snapshot_download failed (exit {result.returncode})", code=result.returncode)


def _fetch_longcat(dest_dir: Path) -> None:
    """Snapshot-download the LongCat-AudioDiT model via the provider's venv."""
    venv_python = repo_root() / "providers" / "longcat-audiodit" / ".venv" / "bin" / "python"
    if not venv_python.exists():
        die(
            f"longcat venv python not found at {venv_python}; "
            "run `mise run install-provider longcat-audiodit --skip-models` first"
        )
    _hf_snapshot(venv_python, LONGCAT_HF_REPO, dest_dir / LONGCAT_DIR_NAME)


MOSS_TTS_NANO_DIR_NAME = "moss-tts-nano"
MOSS_TTS_NANO_HF_REPOS = (
    ("OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX", "tts"),
    ("OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX", "tokenizer"),
)


def _fetch_moss_tts_nano(dest_dir: Path) -> None:
    """Snapshot-download both MOSS-TTS-Nano ONNX repos via the provider's venv."""
    venv_python = repo_root() / "providers" / "moss-tts-nano" / ".venv" / "bin" / "python"
    if not venv_python.exists():
        die(
            f"moss-tts-nano venv python not found at {venv_python}; "
            "run `mise run install-provider moss-tts-nano --skip-models` first"
        )
    target_root = dest_dir / MOSS_TTS_NANO_DIR_NAME
    for repo_id, subdir in MOSS_TTS_NANO_HF_REPOS:
        _hf_snapshot(venv_python, repo_id, target_root / subdir)


OMNIVOICE_HF_REPO = "k2-fsa/OmniVoice"
OMNIVOICE_DIR_NAME = "omnivoice"


def _fetch_omnivoice(dest_dir: Path) -> None:
    """Snapshot-download the OmniVoice model via the provider's venv."""
    venv_python = repo_root() / "providers" / "omnivoice" / ".venv" / "bin" / "python"
    if not venv_python.exists():
        die(
            f"omnivoice venv python not found at {venv_python}; "
            "run `mise run install-provider omnivoice --skip-models` first"
        )
    _hf_snapshot(venv_python, OMNIVOICE_HF_REPO, dest_dir / OMNIVOICE_DIR_NAME)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="download_models.py")
    parser.add_argument("provider", nargs="?", default="kokoro-onnx")
    parser.add_argument(
        "--with-int8",
        action="store_true",
        help="Also download the older INT8 variant (smaller but slower on modern CPUs)",
    )
    # Back-compat: --with-fp16-gpu used to gate the FP16 download. Now it's
    # the default; the flag is a no-op and we just warn so existing scripts
    # don't break.
    parser.add_argument(
        "--with-fp16-gpu",
        action="store_true",
        help="(deprecated, no-op — FP16 is now the default)",
    )
    args = parser.parse_args(argv)
    if args.with_fp16_gpu:
        warn("--with-fp16-gpu is now a no-op; FP16 is the default")

    dest = models_dir()
    if args.provider == "kokoro-onnx":
        _fetch_set(KOKORO_FILES, dest)
        if args.with_int8:
            _fetch_set(KOKORO_INT8_FILES, dest)
    elif args.provider == "longcat-audiodit":
        _fetch_longcat(dest)
    elif args.provider == "moss-tts-nano":
        _fetch_moss_tts_nano(dest)
    elif args.provider == "omnivoice":
        _fetch_omnivoice(dest)
    else:
        die(f"download_models: provider {args.provider!r} has no built-in download spec yet")
    info(f"models in {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
