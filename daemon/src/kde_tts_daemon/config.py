"""Config paths and on-disk daemon settings."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


def _xdg(env: str, default_subpath: str) -> Path:
    base = os.environ.get(env)
    if base:
        return Path(base)
    return Path.home() / default_subpath


XDG_CONFIG_HOME = _xdg("XDG_CONFIG_HOME", ".config")
XDG_DATA_HOME = _xdg("XDG_DATA_HOME", ".local/share")
XDG_RUNTIME_DIR = Path(os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}")

CONFIG_DIR = XDG_CONFIG_HOME / "kde-tts-daemon"
DATA_DIR = XDG_DATA_HOME / "kde-tts-daemon"
MODELS_DIR = DATA_DIR / "models"
VOICES_DIR = DATA_DIR / "voices"

CONFIG_FILE = CONFIG_DIR / "config.toml"

SPEECHD_SOCKET_PATH = XDG_RUNTIME_DIR / "kde-tts.sock"
CONTROL_SOCKET_PATH = XDG_RUNTIME_DIR / "kde-tts-control.sock"

WIRE_SAMPLE_RATE = 24_000


@dataclass
class ProviderConfig:
    default: str = "kokoro-onnx"


@dataclass
class SupervisorConfig:
    idle_timeout_seconds: int = 600
    eager_startup: bool = False


@dataclass
class DaemonConfig:
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)

    @classmethod
    def load(cls, path: Path = CONFIG_FILE) -> "DaemonConfig":
        if not path.exists():
            return cls()
        with path.open("rb") as f:
            raw = tomllib.load(f)
        return cls(
            provider=ProviderConfig(**raw.get("provider", {})),
            supervisor=SupervisorConfig(**raw.get("supervisor", {})),
        )


def repo_root() -> Path:
    """Locate the repository root from the daemon's installed location.

    The daemon is invoked as `<repo>/daemon/.venv/bin/python -m kde_tts_daemon`,
    so we walk up from this file: daemon/src/kde_tts_daemon/config.py -> repo.
    """
    return Path(__file__).resolve().parents[3]


def providers_registry_path() -> Path:
    return repo_root() / "providers" / "registry.toml"
