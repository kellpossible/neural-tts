"""In-memory global voice index.

Maps each voice id to the provider that owns it. Populated lazily on first
LIST VOICES from speechd by spawning each enabled provider in turn (see
Supervisor.enumerate_all_voices). Lives only for the daemon's lifetime —
the next daemon start re-enumerates.

When two providers expose voices with the same id (e.g. both
moss-tts-nano and omnivoice surface a `hobbits` clone), the index
disambiguates them by suffixing the provider name: the colliding voices
become `hobbits.moss-tts-nano` and `hobbits.omnivoice` in the public
listing. The provider itself still only knows the bare id, so the
supervisor translates back via `local_id_for` before sending a synth
request to the provider subprocess.
"""

from __future__ import annotations

import logging
from collections import Counter

from .voices import Voice

log = logging.getLogger("neural_tts_daemon.voice_index")


class VoiceIndex:
    """In-memory index of voices across all enabled providers."""

    def __init__(self, enabled_providers: list[str]) -> None:
        self._enabled = list(enabled_providers)
        # Raw provider output (bare ids, as the provider knows them).
        self._by_provider_raw: dict[str, list[Voice]] = {}
        # Public listing (collision suffixes applied).
        self._by_provider_public: dict[str, list[Voice]] = {}
        # public_id → (provider_name, local_id).
        self._by_public_id: dict[str, tuple[str, str]] = {}

    # ── population ─────────────────────────────────────────────────────

    def set_provider_voices(self, provider: str, voices: list[Voice]) -> None:
        """Replace one provider's slice of the index."""
        self._by_provider_raw[provider] = list(voices)
        self._rebuild()

    def clear(self) -> None:
        self._by_provider_raw.clear()
        self._by_provider_public.clear()
        self._by_public_id.clear()

    def is_empty(self) -> bool:
        return not self._by_public_id

    # ── queries ────────────────────────────────────────────────────────

    def all_voices(self) -> list[Voice]:
        """Union of every provider's voice list with collision suffixes applied,
        in (provider, then voice) order."""
        return [
            v
            for provider in sorted(self._by_provider_public)
            for v in self._by_provider_public[provider]
        ]

    def provider_for(self, public_id: str) -> str | None:
        entry = self._by_public_id.get(public_id)
        return entry[0] if entry else None

    def local_id_for(self, public_id: str) -> str:
        """Translate a (possibly collision-suffixed) public id back to the bare id
        the owning provider knows. Returns `public_id` unchanged if it isn't in
        the index — the caller will discover that via the subsequent lookup."""
        entry = self._by_public_id.get(public_id)
        return entry[1] if entry else public_id

    def voices_for(self, provider: str) -> list[Voice]:
        return list(self._by_provider_public.get(provider, []))

    def known_providers(self) -> list[str]:
        return sorted(self._by_provider_raw)

    # ── internals ──────────────────────────────────────────────────────

    def _rebuild(self) -> None:
        """Recompute the public listing + lookup from `_by_provider_raw`.

        Detection is symmetric: if two or more providers expose the same bare
        id, every copy gets the `<id>.<provider>` suffix. If only one provider
        owns an id, it stays bare.
        """
        id_counts: Counter[str] = Counter()
        for voices in self._by_provider_raw.values():
            for v in voices:
                id_counts[v.id] += 1

        self._by_provider_public = {}
        self._by_public_id = {}
        for provider in sorted(self._by_provider_raw):
            public_list: list[Voice] = []
            for v in self._by_provider_raw[provider]:
                if id_counts[v.id] > 1:
                    public_id = f"{v.id}.{provider}"
                    public_v = Voice(
                        id=public_id,
                        language=v.language,
                        gender=v.gender,
                        display_name=v.display_name,
                        extra_languages=v.extra_languages,
                    )
                    log.info(
                        "voice id %r collision across providers; exposing %s as %r",
                        v.id, provider, public_id,
                    )
                else:
                    public_id = v.id
                    public_v = v
                if public_id in self._by_public_id:
                    other_provider, _ = self._by_public_id[public_id]
                    log.warning(
                        "public voice id %r already claimed by %r; keeping that one",
                        public_id, other_provider,
                    )
                    continue
                self._by_public_id[public_id] = (provider, v.id)
                public_list.append(public_v)
            self._by_provider_public[provider] = public_list
