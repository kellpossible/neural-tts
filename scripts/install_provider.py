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
from _common import (  # noqa: E402
    config_dir,
    die,
    enabled_providers_in_config,
    info,
    load_registry,
    repo_root,
    require,
    warn,
)
from _longcat_pin import LONGCAT_UPSTREAM_COMMIT, LONGCAT_UPSTREAM_REPO  # noqa: E402
from _moss_tts_nano_pin import (  # noqa: E402
    MOSS_TTS_NANO_UPSTREAM_COMMIT,
    MOSS_TTS_NANO_UPSTREAM_REPO,
)


def _vendor_repo(project_dir: Path, *, name: str, repo_url: str, commit: str) -> None:
    """Clone (or fast-forward to) a pinned upstream commit into vendor/<name>/.

    The cloned repo sits next to the provider's .venv. The provider's engine.py
    is responsible for adding its path to sys.path before importing.
    """
    vendor_root = project_dir / "vendor" / name
    git = require("git", "https://git-scm.com/")

    if not (vendor_root / ".git").is_dir():
        vendor_root.parent.mkdir(parents=True, exist_ok=True)
        info(f"cloning upstream {name} into {vendor_root}")
        result = subprocess.run([str(git), "clone", repo_url, str(vendor_root)])
        if result.returncode != 0:
            die(f"git clone failed (exit {result.returncode})", code=result.returncode)

    info(f"checking out pinned commit {commit[:12]}")
    result = subprocess.run([str(git), "fetch", "--quiet", "origin", commit], cwd=vendor_root)
    if result.returncode != 0:
        die(f"git fetch of pinned commit failed (exit {result.returncode})", code=result.returncode)
    result = subprocess.run(
        [str(git), "-c", "advice.detachedHead=false", "checkout", "--quiet", commit],
        cwd=vendor_root,
    )
    if result.returncode != 0:
        die(f"git checkout failed (exit {result.returncode})", code=result.returncode)


def _vendor_longcat(project_dir: Path) -> None:
    _vendor_repo(
        project_dir,
        name="LongCat-AudioDiT",
        repo_url=LONGCAT_UPSTREAM_REPO,
        commit=LONGCAT_UPSTREAM_COMMIT,
    )


def _vendor_moss_tts_nano(project_dir: Path) -> None:
    _vendor_repo(
        project_dir,
        name="MOSS-TTS-Nano",
        repo_url=MOSS_TTS_NANO_UPSTREAM_REPO,
        commit=MOSS_TTS_NANO_UPSTREAM_COMMIT,
    )


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

    # Some providers need upstream source on sys.path. Clone it before
    # `uv sync` so the venv is fully usable as soon as sync finishes.
    if args.provider == "longcat-audiodit":
        _vendor_longcat(project_dir)
    elif args.provider == "moss-tts-nano":
        _vendor_moss_tts_nano(project_dir)

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
        # FP16 is now the kokoro default download — no flag needed for --extra gpu.
        result = subprocess.run(dl_cmd, env=env)
        if result.returncode != 0:
            die(f"download_models failed (exit {result.returncode})", code=result.returncode)

    info(f"provider {args.provider} installed.")
    _warn_if_not_enabled(args.provider)
    info("Next: `mise run install` (if you haven't yet) then `bin/neural-tts-ctl switch " f"{args.provider}` if the daemon is already running.")
    return 0


def _warn_if_not_enabled(provider: str) -> None:
    """If the user has a config.toml already, nudge them to enable the provider.

    We intentionally do NOT edit config.toml — it may contain user comments
    and tomllib is read-only. A clear hint is friendlier than a surprise edit.
    """
    enabled = enabled_providers_in_config()
    if enabled is None:
        # No config yet — `mise run install` will seed one and auto-detect
        # installed providers. Nothing to nudge about here.
        return
    if provider in enabled:
        return
    cfg_path = config_dir() / "config.toml"
    updated = enabled + [provider]
    new_line = "enabled = [" + ", ".join(f'"{n}"' for n in updated) + "]"
    warn(
        f"{provider!r} is installed but NOT in [provider] enabled in {cfg_path}.\n"
        f"     Add it by replacing the existing `enabled = [...]` line with:\n"
        f"         {new_line}"
    )


if __name__ == "__main__":
    sys.exit(main())
