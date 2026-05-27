"""Kokoro ONNX provider — wraps kokoro.Kokoro and exposes pb-shaped values."""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import numpy as np

from .engine import build_kokoro, resolve_model_paths, select_provider
from .pb import neural_tts_pb2 as pb

log = logging.getLogger("neural_tts_provider_kokoro_onnx.provider")

# Mirrors daemon/src/neural_tts_daemon/voices.py: KOKORO_PREFIX_MAP.
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


def _scan_voice_names() -> list[str]:
    """Enumerate kokoro voice ids without instantiating the Kokoro model.

    `voices-v1.0.bin` is a numpy NpzFile whose top-level keys are the voice
    ids. Reading the file header is cheap (no model load, no ORT session).
    Returns [] if the voices file isn't on disk yet.
    """
    _, voices_path = resolve_model_paths(using_gpu=False)
    if not voices_path.exists():
        return []
    try:
        archive = np.load(str(voices_path))
        return sorted(archive.keys())
    except Exception as e:
        log.warning("could not scan %s: %s", voices_path, e)
        return []


class KokoroProvider:
    def __init__(self, eager_startup: bool = False) -> None:
        self._eager = eager_startup
        self._kokoro = None
        self._cached_voice_names: list[str] = []
        self.sample_rate = SAMPLE_RATE

    def _voice_pb(self, voice_id: str) -> pb.Voice:
        prefix = voice_id.split("_", 1)[0]
        lang, gender = _PREFIX_MAP.get(prefix, ("en-US", pb.FEMALE))
        return pb.Voice(id=voice_id, language=lang, gender=gender, display_name=voice_id)

    def list_voices_pb(self) -> list[pb.Voice]:
        # Once the model is loaded, prefer its authoritative list (catches any
        # drift between disk and what the lib will actually accept). Otherwise
        # fall back to the cheap on-disk scan.
        if self._kokoro is not None:
            names = sorted(self._kokoro.get_voices())
        else:
            names = self._cached_voice_names
        return [self._voice_pb(v) for v in names]

    async def warmup(self) -> tuple[int, list[pb.Voice]]:
        """Quick voice + sample-rate enumeration. Model load is deferred to the
        first `synthesize_stream` call unless `eager_startup=True`."""
        self._cached_voice_names = await asyncio.to_thread(_scan_voice_names)
        if not self._cached_voice_names:
            log.warning(
                "no voices found in voices-v1.0.bin — has `mise run download-models kokoro-onnx` been run?"
            )

        if self._eager:
            await self._ensure_model_loaded()
            first = sorted(self._kokoro.get_voices())[0]
            log.info("eager warmup: warming ORT session with voice %s", first)
            async for _samples, _sr in self._kokoro.create_stream(
                "Warming up.", voice=first, speed=1.0, lang="en-us"
            ):
                pass
            log.info("eager warmup complete; %d voices available", len(self._cached_voice_names))
        else:
            log.info(
                "lazy warmup: %d voice(s) enumerated, model load deferred to first synth",
                len(self._cached_voice_names),
            )

        return SAMPLE_RATE, self.list_voices_pb()

    async def _ensure_model_loaded(self) -> None:
        if self._kokoro is not None:
            return
        log.info("loading kokoro model (deferred)")
        self._kokoro = await asyncio.to_thread(build_kokoro)

    async def synthesize_stream(
        self, *, voice: str, speed: float, lang: str, text: str
    ) -> AsyncIterator[np.ndarray]:
        await self._ensure_model_loaded()
        async for samples, _sr in self._kokoro.create_stream(
            text, voice=voice, speed=speed, lang=lang
        ):
            yield np.asarray(samples, dtype=np.float32, copy=False)

    async def shutdown(self) -> None:
        self._kokoro = None
        self._cached_voice_names = []
