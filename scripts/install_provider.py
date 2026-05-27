#!/usr/bin/env python3
"""Install a provider venv: run uv sync in its project dir, optionally fetch models."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import die, info, load_registry, repo_root, require  # noqa: E402
from _longcat_pin import LONGCAT_UPSTREAM_COMMIT, LONGCAT_UPSTREAM_REPO  # noqa: E402


def _vendor_longcat(project_dir: Path) -> None:
    """Clone (or fast-forward to) the pinned LongCat-AudioDiT upstream commit.

    The cloned repo sits next to the provider's .venv. engine.py adds its
    path to sys.path before importing the `audiodit` package.
    """
    vendor_root = project_dir / "vendor" / "LongCat-AudioDiT"
    git = require("git", "https://git-scm.com/")

    if not (vendor_root / ".git").is_dir():
        vendor_root.parent.mkdir(parents=True, exist_ok=True)
        info(f"cloning upstream LongCat-AudioDiT into {vendor_root}")
        result = subprocess.run([str(git), "clone", LONGCAT_UPSTREAM_REPO, str(vendor_root)])
        if result.returncode != 0:
            die(f"git clone failed (exit {result.returncode})", code=result.returncode)

    info(f"checking out pinned commit {LONGCAT_UPSTREAM_COMMIT[:12]}")
    result = subprocess.run([str(git), "fetch", "--quiet", "origin", LONGCAT_UPSTREAM_COMMIT], cwd=vendor_root)
    if result.returncode != 0:
        die(f"git fetch of pinned commit failed (exit {result.returncode})", code=result.returncode)
    result = subprocess.run(
        [str(git), "-c", "advice.detachedHead=false", "checkout", "--quiet", LONGCAT_UPSTREAM_COMMIT],
        cwd=vendor_root,
    )
    if result.returncode != 0:
        die(f"git checkout failed (exit {result.returncode})", code=result.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="install_provider.py")
    parser.add_argument("provider", help="Provider name from providers/registry.toml")
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="uv extras to enable (e.g. --extra gpu). May be repeated.",
    )
    parser.add_argument("--skip-models", action="store_true", help="Don't download model files")
    args = parser.parse_args(argv)

    registry = load_registry()
    if args.provider not in registry:
        die(f"unknown provider {args.provider!r} (known: {sorted(registry)})")
    meta = registry[args.provider]
    project_dir = repo_root() / meta["project_dir"]
    if not project_dir.exists():
        die(f"provider project dir not found: {project_dir}")

    uv = require("uv", "https://docs.astral.sh/uv/")

    # LongCat needs upstream's `audiodit/` source on sys.path. Clone it before
    # `uv sync` so the venv is fully usable as soon as sync finishes.
    if args.provider == "longcat-audiodit":
        _vendor_longcat(project_dir)

    cmd = [str(uv), "sync"]
    for x in args.extra:
        cmd.extend(["--extra", x])
    info(f"syncing {args.provider} venv: {' '.join(cmd)}  (in {project_dir})")
    result = subprocess.run(cmd, cwd=project_dir)
    if result.returncode != 0:
        die(f"uv sync failed (exit {result.returncode})", code=result.returncode)

    venv_python = project_dir / ".venv" / "bin" / "python"
    if not venv_python.exists():
        die(f"venv python not found after sync: {venv_python}")
    info(f"venv ready at {project_dir / '.venv'}")

    if meta.get("needs_models", False) and not args.skip_models:
        info(f"downloading models for {args.provider}")
        env = os.environ.copy()
        dl_cmd = [
            sys.executable,
            str(repo_root() / "scripts" / "download_models.py"),
            args.provider,
        ]
        # `--extra gpu` implies we want the fp16-gpu model (daemon prefers it).
        if args.provider == "kokoro-onnx" and "gpu" in args.extra:
            dl_cmd.append("--with-fp16-gpu")
        result = subprocess.run(dl_cmd, env=env)
        if result.returncode != 0:
            die(f"download_models failed (exit {result.returncode})", code=result.returncode)

    info(f"provider {args.provider} installed.")
    info("Next: `mise run install` (if you haven't yet) then `bin/neural-tts-ctl switch " f"{args.provider}` if the daemon is already running.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
