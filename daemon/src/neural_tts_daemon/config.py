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

CONFIG_DIR = XDG_CONFIG_HOME / "neural-tts-daemon"
DATA_DIR = XDG_DATA_HOME / "neural-tts-daemon"
MODELS_DIR = DATA_DIR / "models"
VOICES_DIR = DATA_DIR / "voices"

CONFIG_FILE = CONFIG_DIR / "config.toml"

SPEECHD_SOCKET_PATH = XDG_RUNTIME_DIR / "neural-tts.sock"
CONTROL_SOCKET_PATH = XDG_RUNTIME_DIR / "neural-tts-control.sock"

WIRE_SAMPLE_RATE = 24_000


@dataclass
class ProviderConfig:
    default: str = "kokoro-onnx"
    # Allowlist of providers the daemon will surface. Providers not listed
    # here are invisible — `switch` rejects them, `known_providers` omits
    # them. An empty list means nothing is enabled; the daemon will refuse
    # to spawn until the user adds at least one provider here.
    enabled: list[str] = field(default_factory=list)


@dataclass
class SupervisorConfig:
    idle_timeout_seconds: int = 600
    eager_startup: bool = False


@dataclass
class ProviderSettings:
    """Per-provider settings from `[providers.<name>]` tables.

    `voices` is an optional allowlist of voice ids; when set, the daemon
    drops any other voices from that provider during enumeration. Absent or
    empty means "no filtering — surface every voice the provider reports".

    `env` is a dict of environment variables to inject into the provider
    subprocess. Use it to set knobs the provider already reads from its
    environment (e.g. `TTS_OMNIVOICE_NUM_STEP`, `TTS_KOKORO_MODEL_PATH`).
    Values must be strings — TOML's typed values are coerced to str before
    being passed to the subprocess. Pre-existing env vars in the daemon's
    own environment are overridden by this map.
    """
    voices: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    # Voice-id glob → list of BCP-47 tags. When a voice id matches, the
    # provider-declared language is *replaced* with this list (first tag
    # becomes the primary, the rest are advertised as additional locales).
    # Declaration order in TOML decides precedence — first matching glob wins.
    locales: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class DaemonConfig:
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)
    # provider_name → settings. Keyed by the same names used in the registry.
    providers: dict[str, ProviderSettings] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = CONFIG_FILE) -> "DaemonConfig":
        if not path.exists():
            return cls()
        with path.open("rb") as f:
            raw = tomllib.load(f)
        providers_raw = raw.get("providers", {}) or {}
        providers: dict[str, ProviderSettings] = {}
        for name, settings in providers_raw.items():
            raw_env = settings.get("env") or {}
            raw_locales = settings.get("locales") or {}
            locales: dict[str, list[str]] = {}
            for pattern, tags in raw_locales.items():
                if isinstance(tags, str):
                    tags = [tags]
                locales[str(pattern)] = [str(t) for t in tags if str(t).strip()]
            providers[name] = ProviderSettings(
                voices=list(settings.get("voices") or []),
                env={str(k): str(v) for k, v in raw_env.items()},
                locales=locales,
            )
        return cls(
            provider=ProviderConfig(**raw.get("provider", {})),
            supervisor=SupervisorConfig(**raw.get("supervisor", {})),
            providers=providers,
        )


def repo_root() -> Path:
    """Locate the repository root from the daemon's installed location.

    The daemon is invoked as `<repo>/daemon/.venv/bin/python -m neural_tts_daemon`,
    so we walk up from this file: daemon/src/neural_tts_daemon/config.py -> repo.
    """
    return Path(__file__).resolve().parents[3]


def providers_registry_path() -> Path:
    return repo_root() / "providers" / "registry.toml"
