"""pocket-tts provider.

Implements warmup / list_voices / synthesize_stream / shutdown against the
neural-tts-daemon provider protocol. Surfaces pocket-tts' built-in preset
voices plus user-supplied cloned clips (see voices.py). Synthesis streams
pocket-tts' `generate_audio_stream` (a blocking generator) through a thread +
asyncio.Queue bridge, yielding mono float32 PCM as each chunk is decoded.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np

from . import voices as voices_mod
from .engine import NATIVE_SAMPLE_RATE, build_model
from .pb import neural_tts_pb2 as pb
from .voices import VoiceEntry

log = logging.getLogger("neural_tts_provider_pocket_tts.provider")


def _clone_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "neural-tts-daemon" / "pocket-tts" / "clones"


class PocketTtsProvider:
    def __init__(self, eager_startup: bool = False) -> None:
        self._eager = eager_startup
        self._model: Any | None = None
        self._voices: dict[str, VoiceEntry] = {}
        # voice_id → (source_mtime_ns, voice_state). mtime is 0 for presets.
        self._state_cache: dict[str, tuple[int, Any]] = {}
        self.sample_rate = NATIVE_SAMPLE_RATE

    # ── voice management ───────────────────────────────────────────────

    def _rescan_voices(self) -> None:
        entries = voices_mod.preset_entries() + voices_mod.scan_cloned_voices()
        self._voices = {e.voice_id: e for e in entries}
        # Drop cached states for voices that vanished.
        for stale in [k for k in self._state_cache if k not in self._voices]:
            self._state_cache.pop(stale, None)

    def list_voices_pb(self) -> list[pb.Voice]:
        self._rescan_voices()
        return [voices_mod.to_pb(v) for v in self._voices.values()]

    # ── lifecycle ──────────────────────────────────────────────────────

    async def warmup(self) -> tuple[int, list[pb.Voice]]:
        """Enumerate voices (cheap). Model load is deferred to the first
        synthesize call unless eager_startup=True."""
        self._rescan_voices()
        if self._eager:
            await self._ensure_model_loaded()
            first = next(iter(self._voices.values()))
            log.info("eager warmup: warming model with voice %s", first.voice_id)
            try:
                async for _pcm in self.synthesize_stream(
                    voice=first.voice_id, speed=1.0, lang=first.language, text="Warming up."
                ):
                    pass
            except Exception:
                log.exception("eager warmup synth failed (model loaded, continuing)")
        else:
            log.info(
                "lazy warmup: %d voice(s) enumerated, model load deferred to first synth",
                len(self._voices),
            )
        return self.sample_rate, self.list_voices_pb()

    async def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        log.info("loading pocket-tts model (deferred)")
        self._model = await asyncio.to_thread(build_model)
        if getattr(self._model, "has_voice_cloning", True):
            log.info("voice cloning model available (gated kyutai/pocket-tts loaded)")
        else:
            log.info(
                "voice cloning unavailable (gated model not installed); "
                "preset voices only — cloned clips will be rejected at synth time"
            )
        actual = int(self._model.sample_rate)
        if actual != self.sample_rate:
            log.warning(
                "model sample_rate differs from assumed constant: %d→%d",
                self.sample_rate, actual,
            )
            self.sample_rate = actual

    async def shutdown(self) -> None:
        log.info("shutting down pocket-tts provider")
        self._voices.clear()
        self._state_cache.clear()
        self._model = None

    # ── synthesis ──────────────────────────────────────────────────────

    async def synthesize_stream(
        self, *, voice: str, speed: float, lang: str, text: str
    ) -> AsyncIterator[np.ndarray]:
        await self._ensure_model_loaded()
        if voice not in self._voices:
            raise RuntimeError(
                f"unknown voice {voice!r}; available: {sorted(self._voices) or 'NONE'}"
            )
        entry = self._voices[voice]

        if abs(speed - 1.0) > 0.05:
            log.warning("pocket-tts has no speed knob; ignoring speed=%.2f", speed)

        text = text.strip()
        if not text:
            return

        state = await asyncio.to_thread(self._get_voice_state, entry)
        async for pcm in self._stream(state, text):
            if pcm.size:
                yield pcm

    # ── streaming bridge: sync generator thread → async iterator ───────

    async def _stream(self, state: Any, text: str) -> AsyncIterator[np.ndarray]:
        """Drive `generate_audio_stream` in a thread; yield PCM as it arrives.

        pocket-tts' generator is blocking and yields 1-D torch tensors
        (audio_chunk[0, 0]); we convert to float32 numpy on the producer thread.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[np.ndarray | object] = asyncio.Queue()
        sentinel: object = object()

        def producer() -> None:
            try:
                for chunk in self._model.generate_audio_stream(state, text):
                    tensor = chunk.detach().to("cpu") if hasattr(chunk, "detach") else chunk
                    arr = np.asarray(tensor, dtype=np.float32).reshape(-1)
                    loop.call_soon_threadsafe(queue.put_nowait, arr)
            except BaseException as e:  # noqa: BLE001 — re-raised on the consumer
                loop.call_soon_threadsafe(queue.put_nowait, e)
            else:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        task = asyncio.create_task(asyncio.to_thread(producer))
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item  # type: ignore[misc]
        finally:
            await task

    # ── voice-state resolution (blocking; run inside a thread) ─────────

    def _get_voice_state(self, entry: VoiceEntry) -> Any:
        """Return the pocket-tts voice state for `entry`, computing + caching it.

        Presets resolve from pocket-tts' bundled safetensors (cheap, cached in
        HF). Cloned clips are encoded from the wav (slow) and persisted as
        safetensors keyed by the source mtime so a restart reuses them.
        """
        assert self._model is not None

        # Preset: id is accepted directly by get_state_for_audio_prompt.
        if entry.wav_path is None:
            cached = self._state_cache.get(entry.voice_id)
            if cached is not None:
                return cached[1]
            log.info("resolving preset voice state: %s", entry.voice_id)
            state = self._model.get_state_for_audio_prompt(entry.voice_id)
            self._state_cache[entry.voice_id] = (0, state)
            return state

        # Clone: cache in memory + on disk, keyed by the wav's mtime.
        mtime_ns = entry.wav_path.stat().st_mtime_ns
        cached = self._state_cache.get(entry.voice_id)
        if cached is not None and cached[0] == mtime_ns:
            return cached[1]

        from pocket_tts import export_model_state

        cache_dir = _clone_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{entry.voice_id}.{mtime_ns}.safetensors"

        if cache_file.exists():
            log.info("loading cached voice state for %s from %s", entry.voice_id, cache_file.name)
            state = self._model.get_state_for_audio_prompt(str(cache_file))
        else:
            # Encoding a fresh clip needs the gated cloning model; a cached
            # .safetensors state above imports fine without it.
            if not getattr(self._model, "has_voice_cloning", True):
                raise RuntimeError(
                    f"voice cloning for {entry.voice_id!r} needs the gated "
                    "kyutai/pocket-tts model, which is not installed. Accept the terms "
                    "at https://huggingface.co/kyutai/pocket-tts, run `hf auth login`, "
                    "then re-run `mise run download-models pocket-tts`."
                )
            log.info("encoding reference audio for %s from %s", entry.voice_id, entry.wav_path)
            state = self._model.get_state_for_audio_prompt(str(entry.wav_path))
            # Drop stale cache files for this voice, then persist the fresh one.
            for old in cache_dir.glob(f"{entry.voice_id}.*.safetensors"):
                try:
                    old.unlink()
                except OSError:
                    pass
            try:
                export_model_state(state, str(cache_file))
            except Exception:
                log.warning("could not persist voice-state cache for %s", entry.voice_id, exc_info=True)

        self._state_cache[entry.voice_id] = (mtime_ns, state)
        return state
