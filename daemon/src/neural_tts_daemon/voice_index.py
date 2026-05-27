"""Persistent global voice index.

Maps each voice id to the provider that owns it. Populated lazily on first
LIST VOICES from speechd by spawning each enabled provider in turn (see
Supervisor.enumerate_all_voices), then persisted to disk so future daemon
starts skip the spawn-and-list cost.

Cache invalidation: the header records the set of enabled providers and a
schema version. A mismatch on load means we throw away the cache and
re-enumerate on the next list-voices request.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .voices import Voice

log = logging.getLogger("neural_tts_daemon.voice_index")

SCHEMA_VERSION = 1


def _default_cache_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "neural-tts-daemon" / "voice_index.json"


class VoiceIndex:
    """In-memory + on-disk index of voices across all enabled providers."""

    def __init__(self, enabled_providers: list[str], cache_path: Path | None = None) -> None:
        self._enabled = list(enabled_providers)
        self._path = cache_path or _default_cache_path()
        # provider_name → [Voice, ...]
        self._by_provider: dict[str, list[Voice]] = {}
        # voice_id → (provider_name, Voice). Built lazily from _by_provider.
        self._by_voice: dict[str, tuple[str, Voice]] = {}
        self._loaded_from_disk = False

    # ── lifecycle ──────────────────────────────────────────────────────

    def load(self) -> None:
        """Read the on-disk cache; silently start empty if missing or stale.

        "Stale" = schema version mismatch OR the enabled-providers set changed
        since the cache was written.
        """
        if not self._path.exists():
            log.info("voice index: no cache at %s — will populate on first use", self._path)
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            log.warning("voice index: cache unreadable (%s) — discarding", e)
            return
        if raw.get("schema_version") != SCHEMA_VERSION:
            log.info("voice index: schema mismatch — discarding cache")
            return
        cached_enabled = sorted(raw.get("enabled_providers") or [])
        if cached_enabled != sorted(self._enabled):
            log.info(
                "voice index: enabled-providers changed (cache=%s, current=%s) — discarding cache",
                cached_enabled, sorted(self._enabled),
            )
            return
        providers = raw.get("providers") or {}
        for name, voices_raw in providers.items():
            self._by_provider[name] = [Voice.from_json(v) for v in voices_raw]
        self._rebuild_voice_map()
        self._loaded_from_disk = True
        log.info(
            "voice index: loaded %d voice(s) across %d provider(s) from %s",
            len(self._by_voice), len(self._by_provider), self._path,
        )

    def save(self) -> None:
        """Atomically write the current index to disk."""
        payload = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "enabled_providers": sorted(self._enabled),
            "providers": {
                name: [v.to_json() for v in voices]
                for name, voices in sorted(self._by_provider.items())
            },
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write via tempfile + rename. Keeps the cache file from being
        # truncated if we crash mid-write.
        fd, tmp = tempfile.mkstemp(prefix=".voice_index.", suffix=".json", dir=str(self._path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, self._path)
        except Exception:
            try: os.unlink(tmp)
            except OSError: pass
            raise
        log.info("voice index: saved %d voice(s) to %s", len(self._by_voice), self._path)

    # ── population ─────────────────────────────────────────────────────

    def set_provider_voices(self, provider: str, voices: list[Voice]) -> None:
        """Replace one provider's slice of the index."""
        self._by_provider[provider] = list(voices)
        self._rebuild_voice_map()

    def clear(self) -> None:
        self._by_provider.clear()
        self._by_voice.clear()

    def is_empty(self) -> bool:
        return not self._by_voice

    # ── queries ────────────────────────────────────────────────────────

    def all_voices(self) -> list[Voice]:
        """Union of every provider's voice list, in (provider, then voice) order."""
        return [v for provider in sorted(self._by_provider)
                for v in self._by_provider[provider]]

    def provider_for(self, voice_id: str) -> str | None:
        entry = self._by_voice.get(voice_id)
        return entry[0] if entry else None

    def voices_for(self, provider: str) -> list[Voice]:
        return list(self._by_provider.get(provider, []))

    def known_providers(self) -> list[str]:
        return sorted(self._by_provider)

    # ── internals ──────────────────────────────────────────────────────

    def _rebuild_voice_map(self) -> None:
        """Flatten _by_provider into _by_voice. Warn on cross-provider id collisions.

        Resolution rule: first provider (sorted alphabetically) wins. The user can
        fix collisions by renaming voices in the losing provider.
        """
        self._by_voice = {}
        for provider in sorted(self._by_provider):
            for voice in self._by_provider[provider]:
                if voice.id in self._by_voice:
                    other_provider, _ = self._by_voice[voice.id]
                    log.warning(
                        "voice id %r appears in both %r and %r — keeping %r",
                        voice.id, other_provider, provider, other_provider,
                    )
                    continue
                self._by_voice[voice.id] = (provider, voice)
