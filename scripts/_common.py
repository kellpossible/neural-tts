"""Shared helpers for install/uninstall/download scripts. Stdlib only."""

from __future__ import annotations

import os
import shutil
import sys
import tomllib
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def xdg_config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")


def xdg_data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")


def data_dir() -> Path:
    return xdg_data_home() / "kde-tts-daemon"


def models_dir() -> Path:
    return data_dir() / "models"


def config_dir() -> Path:
    return xdg_config_home() / "kde-tts-daemon"


def systemd_user_dir() -> Path:
    return xdg_config_home() / "systemd" / "user"


def speechd_config_dir() -> Path:
    return xdg_config_home() / "speech-dispatcher"


def speechd_modules_dir() -> Path:
    return speechd_config_dir() / "modules"


def speechd_user_conf() -> Path:
    return speechd_config_dir() / "speechd.conf"


def speechd_system_conf() -> Path:
    return Path("/etc/speech-dispatcher/speechd.conf")


def daemon_python() -> Path:
    return repo_root() / "daemon" / ".venv" / "bin" / "python"


def load_registry() -> dict[str, dict]:
    with (repo_root() / "providers" / "registry.toml").open("rb") as f:
        return tomllib.load(f).get("providers", {})


def require(cmd: str, install_hint: str = "") -> Path:
    p = shutil.which(cmd)
    if not p:
        msg = f"required command not found on PATH: {cmd}"
        if install_hint:
            msg += f"  ({install_hint})"
        die(msg)
    return Path(p)


def die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def info(msg: str) -> None:
    print(f"==> {msg}")


def warn(msg: str) -> None:
    print(f"warn: {msg}", file=sys.stderr)


def confirm(prompt: str, default_no: bool = True, assume_yes: bool = False) -> bool:
    if assume_yes:
        print(f"{prompt} [y]")
        return True
    if not sys.stdin.isatty():
        decision = not default_no
        print(f"{prompt} [{'y' if decision else 'n'}]  (non-interactive)")
        return decision
    suffix = " [y/N] " if default_no else " [Y/n] "
    try:
        ans = input(prompt + suffix).strip().lower()
    except EOFError:
        return False
    if not ans:
        return not default_no
    return ans in ("y", "yes")


def render_template(template: str, **subs: str) -> str:
    out = template
    for key, value in subs.items():
        out = out.replace(f"@{key}@", value)
    return out
