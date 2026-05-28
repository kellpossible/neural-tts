"""In-memory global voice index.

Maps each voice id to the provider that owns it. Populated lazily on first
LIST VOICES from speechd by spawning each enabled provider in turn (see
Supervisor.enumerate_all_voices). Lives only for the daemon's lifetime —
the next daemon start re-enumerates.
"""

from __future__ import annotations

import logging

from .voices import Voice

log = logging.getLogger("neural_tts_daemon.voice_index")


class VoiceIndex:
    """In-memory index of voices across all enabled providers."""

    def __init__(self, enabled_providers: list[str]) -> None:
        self._enabled = list(enabled_providers)
        # provider_name → [Voice, ...]
        self._by_provider: dict[str, list[Voice]] = {}
        # voice_id → (provider_name, Voice). Built lazily from _by_provider.
        self._by_voice: dict[str, tuple[str, Voice]] = {}

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
