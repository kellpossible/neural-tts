#!/usr/bin/env python3
"""Reset a provider: remove its .venv and vendored upstream source.

By default leaves downloaded model files in place (they're large; redownload
is slow). Pass `--purge-models` to also remove them. NEVER touches user
voices or the daemon config — those are explicitly user data.

Usage:
  mise run uninstall-provider <name> [--purge-models] [--yes]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    confirm,
    config_dir,
    die,
    enabled_providers_in_config,
    info,
    load_registry,
    models_dir,
    repo_root,
    warn,
)

# Per-provider knowledge of what `download_models.py` placed under models/.
# Keep in sync with that script. Files are loose at the models root; dirs are
# subdirs we own entirely.
MODEL_ARTEFACTS: dict[str, dict[str, list[str]]] = {
    "kokoro-onnx": {
        "files": [
            "kokoro-v1.0.int8.onnx",
            "kokoro-v1.0.fp16-gpu.onnx",
            "voices-v1.0.bin",
        ],
        "dirs": [],
    },
    "longcat-audiodit": {
        "files": [],
        "dirs": ["longcat-audiodit-1b"],
    },
    "moss-tts-nano": {
        "files": [],
        "dirs": ["moss-tts-nano"],
    },
}


def _remove_path(p: Path) -> None:
    if p.is_symlink() or p.is_file():
        p.unlink()
        info(f"  removed {p}")
    elif p.is_dir():
        shutil.rmtree(p)
        info(f"  removed {p}/")


def _purge_models(provider: str) -> None:
    spec = MODEL_ARTEFACTS.get(provider)
    if spec is None:
        warn(f"no model-artefact spec known for {provider!r}; skipping model purge")
        return
    root = models_dir()
    for name in spec["files"]:
        p = root / name
        if p.exists():
            _remove_path(p)
    for name in spec["dirs"]:
        p = root / name
        if p.exists():
            _remove_path(p)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="uninstall_provider.py")
    parser.add_argument("provider", help="Provider name from providers/registry.toml")
    parser.add_argument(
        "--purge-models",
        action="store_true",
        help="Also delete this provider's downloaded model files",
    )
    parser.add_argument(
        "--yes", action="store_true", help="Assume yes for all prompts (non-interactive)",
    )
    args = parser.parse_args(argv)

    registry = load_registry()
    if args.provider not in registry:
        die(f"unknown provider {args.provider!r} (known: {sorted(registry)})")
    project_dir = repo_root() / registry[args.provider]["project_dir"]

    venv = project_dir / ".venv"
    vendor = project_dir / "vendor"

    targets: list[Path] = [t for t in (venv, vendor) if t.exists()]
    if args.purge_models:
        info(f"will also purge model files for {args.provider}")
    if not targets and not args.purge_models:
        info(f"nothing to remove for {args.provider} (no venv, no vendor dir)")
    else:
        info(f"about to remove for {args.provider}:")
        for t in targets:
            info(f"  - {t}")
        if not confirm("proceed?", default_no=True, assume_yes=args.yes):
            info("aborted; nothing changed")
            return 0
        for t in targets:
            _remove_path(t)

    if args.purge_models:
        _purge_models(args.provider)

    enabled = enabled_providers_in_config()
    if enabled is not None and args.provider in enabled:
        cfg = config_dir() / "config.toml"
        warn(
            f"{args.provider!r} is still listed in [provider] enabled in {cfg}.\n"
            "     Remove it manually if you don't intend to reinstall:\n"
            f"         enabled = [{', '.join(repr(n) for n in enabled if n != args.provider)}]"
        )

    info(f"provider {args.provider} reset.")
    info(f"Reinstall with: mise run install-provider {args.provider}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
