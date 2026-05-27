"""Kokoro ONNX provider — wraps kokoro.Kokoro and exposes pb-shaped values."""

from __future__ import annotations

import logging
from typing import AsyncIterator

import numpy as np

from .engine import build_kokoro
from .pb import kde_tts_pb2 as pb

log = logging.getLogger("kde_tts_provider_kokoro_onnx.provider")

# Mirrors daemon/src/kde_tts_daemon/voices.py: KOKORO_PREFIX_MAP.
# Duplicated to avoid cross-venv imports — if you change one, change the other.
_PREFIX_MAP: dict[str, tuple[str, int]] = {
    "af": ("en-US", pb.FEMALE), "am": ("en-US", pb.MALE),
    "bf": ("en-GB", pb.FEMALE), "bm": ("en-GB", pb.MALE),
    "jf": ("ja", pb.FEMALE), "jm": ("ja", pb.MALE),
    "zf": ("zh", pb.FEMALE), "zm": ("zh", pb.MALE),
    "ef": ("es", pb.FEMALE), "em": ("es", pb.MALE),
    "ff": ("fr", pb.FEMALE), "fm": ("fr", pb.MALE),
    "hf": ("hi", pb.FEMALE), "hm": ("hi", pb.MALE),
    "if": ("it", pb.FEMALE), "im": ("it", pb.MALE),
    "pf": ("pt-BR", pb.FEMALE), "pm": ("pt-BR", pb.MALE),
}

SAMPLE_RATE = 24_000


class KokoroProvider:
    def __init__(self) -> None:
        self._kokoro = None
        self.sample_rate = SAMPLE_RATE

    def _voice_pb(self, voice_id: str) -> pb.Voice:
        prefix = voice_id.split("_", 1)[0]
        lang, gender = _PREFIX_MAP.get(prefix, ("en-US", pb.FEMALE))
        return pb.Voice(id=voice_id, language=lang, gender=gender, display_name=voice_id)

    def list_voices_pb(self) -> list[pb.Voice]:
        if self._kokoro is None:
            return []
        return [self._voice_pb(v) for v in self._kokoro.get_voices()]

    async def warmup(self) -> tuple[int, list[pb.Voice]]:
        log.info("loading kokoro model")
        self._kokoro = build_kokoro()
        voices = self._kokoro.get_voices()
        if not voices:
            raise RuntimeError("kokoro reported zero voices")
        first = sorted(voices)[0]
        log.info("warming up ORT session with voice %s", first)
        async for _samples, _sr in self._kokoro.create_stream(
            "Warming up.", voice=first, speed=1.0, lang="en-us"
        ):
            pass
        log.info("warmup complete; %d voices available", len(voices))
        return SAMPLE_RATE, self.list_voices_pb()

    async def synthesize_stream(
        self, *, voice: str, speed: float, lang: str, text: str
    ) -> AsyncIterator[np.ndarray]:
        if self._kokoro is None:
            raise RuntimeError("provider not warmed up")
        async for samples, _sr in self._kokoro.create_stream(
            text, voice=voice, speed=speed, lang=lang
        ):
            yield np.asarray(samples, dtype=np.float32, copy=False)

    async def shutdown(self) -> None:
        self._kokoro = None
