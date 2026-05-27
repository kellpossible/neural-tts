"""MOSS-TTS-Nano provider.

Implements warmup / list_voices / synthesize_stream / shutdown against the
neural-tts-daemon provider protocol. Voices are user-supplied reference clips
(see voices.py — wav-only, no transcript needed). Text is chunked by the
upstream runtime's own `split_voice_clone_text` (token-budget aware); within
each text-chunk we drive the autoregressive decode ourselves and yield mono
float32 PCM as the codec streaming session emits it — token-level streaming,
not chunk-level.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Callable

import numpy as np

from . import voices as voices_mod
from .engine import build_runtime
from .pb import neural_tts_pb2 as pb
from .voices import VoiceEntry

log = logging.getLogger("neural_tts_provider_moss_tts_nano.provider")

# Upstream default. The runtime splits long text into ~75-token chunks before
# autoregressive decode. Larger ⇒ fewer chunk boundaries (slightly better
# prosody continuity) but more time-to-first-audio per chunk.
VOICE_CLONE_MAX_TOKENS = 75

# Native sample rate / channel count of MOSS-TTS-Nano's codec. These are
# baked into the model and never change — used to populate the WarmupResponse
# in lazy mode without instantiating the ONNX runtime.
NATIVE_SAMPLE_RATE = 48_000
NATIVE_CHANNELS = 2


class MossTtsNanoProvider:
    def __init__(self, eager_startup: bool = False) -> None:
        self._eager = eager_startup
        self._runtime: Any | None = None
        self._voices: dict[str, VoiceEntry] = {}
        # voice_id → prompt_audio_codes (list[list[int]] from
        # runtime.encode_reference_audio). Loaded lazily on first use.
        self._prompt_cache: dict[str, list[list[int]]] = {}
        self.sample_rate = NATIVE_SAMPLE_RATE
        self._channels = NATIVE_CHANNELS

    # ── voice management ───────────────────────────────────────────────

    def _rescan_voices(self) -> None:
        entries = voices_mod.scan_voices()
        self._voices = {e.voice_id: e for e in entries}
        for stale in [k for k in self._prompt_cache if k not in self._voices]:
            self._prompt_cache.pop(stale, None)

    def list_voices_pb(self) -> list[pb.Voice]:
        self._rescan_voices()
        return [voices_mod.to_pb(v) for v in self._voices.values()]

    # ── lifecycle ──────────────────────────────────────────────────────

    async def warmup(self) -> tuple[int, list[pb.Voice]]:
        """Quick voice + sample-rate enumeration. Model load is deferred to the
        first `synthesize_stream` call unless `eager_startup=True`."""
        self._rescan_voices()
        if self._eager:
            await self._ensure_runtime_loaded()
            if self._voices:
                first = next(iter(self._voices.values()))
                log.info("eager warmup: warming ONNX sessions with voice %s", first.voice_id)
                try:
                    await asyncio.to_thread(self._synth_one, first, "Warming up.")
                except Exception:
                    log.exception("eager warmup synth failed (model loaded, continuing)")
        else:
            log.info(
                "lazy warmup: %d voice(s) enumerated, model load deferred to first synth",
                len(self._voices),
            )

        if not self._voices:
            log.warning(
                "no voices found — synthesise requests will fail until reference clips "
                "are dropped into ~/.local/share/neural-tts-daemon/voices/moss-tts-nano/ "
                "and the daemon is asked to reload voices"
            )

        return self.sample_rate, self.list_voices_pb()

    async def _ensure_runtime_loaded(self) -> None:
        """Build the ONNX runtime if it hasn't been built yet."""
        if self._runtime is not None:
            return
        log.info("loading MOSS-TTS-Nano (deferred)")
        self._runtime = await asyncio.to_thread(build_runtime)
        # Sanity-check our hardcoded constants against what the actual codec
        # reports — protects against an upstream model change we missed.
        actual_sr = int(self._runtime.codec_meta["codec_config"]["sample_rate"])
        actual_ch = int(self._runtime.codec_meta["codec_config"]["channels"])
        if actual_sr != self.sample_rate or actual_ch != self._channels:
            log.warning(
                "codec config differs from assumed constants: sr %d→%d, ch %d→%d",
                self.sample_rate, actual_sr, self._channels, actual_ch,
            )
            self.sample_rate = actual_sr
            self._channels = actual_ch

    async def shutdown(self) -> None:
        log.info("shutting down MOSS-TTS-Nano provider")
        self._voices.clear()
        self._prompt_cache.clear()
        self._runtime = None

    # ── synthesis ──────────────────────────────────────────────────────

    async def synthesize_stream(
        self, *, voice: str, speed: float, lang: str, text: str
    ) -> AsyncIterator[np.ndarray]:
        await self._ensure_runtime_loaded()
        if voice not in self._voices:
            raise RuntimeError(
                f"unknown voice {voice!r}; available: "
                f"{sorted(self._voices) or 'NONE — drop clips into the voices/moss-tts-nano dir'}"
            )
        entry = self._voices[voice]

        if abs(speed - 1.0) > 0.05:
            log.warning("MOSS-TTS-Nano has no speed knob; ignoring speed=%.2f", speed)

        # Prepare normalized text (WeTextProcessing intentionally disabled —
        # see pyproject.toml for the rationale).
        prepared = await asyncio.to_thread(
            self._runtime.prepare_synthesis_text,
            text=text,
            voice="",
            enable_wetext=False,
            enable_normalize_tts_text=True,
        )
        prepared_text = str(prepared["text"])
        if not prepared_text:
            return

        chunks: list[str] = await asyncio.to_thread(
            self._runtime.split_voice_clone_text,
            prepared_text,
            VOICE_CLONE_MAX_TOKENS,
        )
        if not chunks:
            return
        log.info(
            "synth voice=%s lang=%s chunks=%d total_chars=%d",
            voice, entry.lang, len(chunks), sum(len(c) for c in chunks),
        )

        prompt_codes = await asyncio.to_thread(self._get_prompt_codes, entry)

        for i, chunk_text in enumerate(chunks):
            log.info("  chunk %d/%d (%d chars): %r", i + 1, len(chunks), len(chunk_text), chunk_text[:60])
            async for pcm in self._stream_chunk(prompt_codes, chunk_text):
                if pcm.size:
                    yield pcm

    # ── streaming bridge: sync producer thread → async iterator ────────

    async def _stream_chunk(
        self, prompt_codes: list[list[int]], text: str
    ) -> AsyncIterator[np.ndarray]:
        """Drive one upstream text-chunk's decode; yield mono PCM as it arrives.

        The autoregressive decode is a blocking call (`generate_audio_frames`),
        so we run it in a thread and pipe its per-frame codec output through an
        asyncio.Queue.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[np.ndarray | object] = asyncio.Queue()
        sentinel: object = object()

        def emit(pcm: np.ndarray) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, pcm)

        def producer() -> None:
            try:
                self._run_streaming_chunk(prompt_codes, text, emit)
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
            # Wait for the producer to settle so the runtime's codec session
            # isn't torn down mid-decode by a later request.
            await task

    # ── synchronous helpers (run inside a thread) ──────────────────────

    def _synth_one(self, entry: VoiceEntry, text: str) -> np.ndarray:
        """Warmup helper: one short synthesis. Accumulates streamed PCM."""
        prompt_codes = self._get_prompt_codes(entry)
        out: list[np.ndarray] = []
        self._run_streaming_chunk(prompt_codes, text, out.append)
        if not out:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(out)

    def _run_streaming_chunk(
        self,
        prompt_codes: list[list[int]],
        text: str,
        emit: Callable[[np.ndarray], None],
    ) -> None:
        """Token-level streaming decode for one upstream text-chunk.

        Replicates upstream `OnnxTtsRuntime.synthesize_single_chunk(streaming=True)`
        (onnx_tts_runtime.py:527-) but calls `emit(pcm)` for each codec
        sub-chunk instead of accumulating into a single waveform. Mono
        downmix happens here so the caller never sees stereo.
        """
        rt = self._runtime
        assert rt is not None

        # `_resolve_stream_decode_frame_budget` is module-private to
        # upstream; we import via the vendor sys.path entry already set up
        # by engine._ensure_vendor_on_path().
        from ort_cpu_runtime import _resolve_stream_decode_frame_budget  # type: ignore[import-not-found]

        text_token_ids = rt.encode_text(text)
        request_rows = rt.build_voice_clone_request_rows(prompt_codes, text_token_ids)
        sample_rate = int(rt.codec_meta["codec_config"]["sample_rate"])

        pending: list[list[int]] = []
        emitted_samples_total = 0
        first_audio_at: float | None = None

        rt.codec_streaming_session.reset()

        def flush(force: bool) -> None:
            nonlocal emitted_samples_total, first_audio_at
            pending_count = len(pending)
            if pending_count <= 0:
                return
            decode_budget = _resolve_stream_decode_frame_budget(
                emitted_samples_total, sample_rate, first_audio_at,
            )
            if not force and pending_count < max(1, decode_budget):
                return
            frame_budget = pending_count if force else min(pending_count, max(1, decode_budget))
            frame_chunk = pending[:frame_budget]
            del pending[:frame_budget]
            decoded = rt.codec_streaming_session.run_frames(frame_chunk)
            if decoded is None:
                return
            audio, audio_length = decoded
            if audio_length <= 0:
                return
            if first_audio_at is None:
                first_audio_at = time.perf_counter()
            emitted_samples_total += audio_length

            # audio shape: (1, channels, samples). Downmix to mono.
            channels = audio.shape[1]
            if channels == 1:
                mono = np.asarray(audio[0, 0, :audio_length], dtype=np.float32)
            else:
                stacked = np.stack(
                    [np.asarray(audio[0, c, :audio_length], dtype=np.float32) for c in range(channels)],
                    axis=1,
                )
                mono = stacked.mean(axis=1)
            emit(mono)

        def on_frame(_generated_frames: list[list[int]], _step_index: int, frame: list[int]) -> None:
            pending.append(list(frame))
            flush(False)

        try:
            rt.generate_audio_frames(request_rows, on_frame=on_frame)
            flush(True)
        finally:
            rt.codec_streaming_session.reset()

    def _get_prompt_codes(self, entry: VoiceEntry) -> list[list[int]]:
        """Encode + cache the reference-audio codes for a voice."""
        cached = self._prompt_cache.get(entry.voice_id)
        if cached is not None:
            return cached
        assert self._runtime is not None
        log.info("encoding reference audio for voice %s from %s", entry.voice_id, entry.wav_path)
        codes = self._runtime.encode_reference_audio(str(entry.wav_path))
        self._prompt_cache[entry.voice_id] = codes
        return codes
