"""Qwen3-TTS provider (faster-qwen3-tts backend).

Implements warmup / list_voices / synthesize_stream / shutdown against the
neural-tts-daemon provider protocol. Voices are user-supplied reference
clips with transcripts (see voices.py).

Streaming: drives `FasterQwen3TTS.generate_voice_clone_streaming(...)`
and forwards each yielded PCM chunk to the daemon. The wrapper streams
natively from the codec, so we hand it the whole utterance — no text
chunker.

Voice-prompt caching: faster-qwen3-tts keeps an internal
`_voice_prompt_cache` keyed by (ref_audio, ref_text), so simply passing
the same paths/text reuses the precomputed ref_code + speaker embedding.
We don't need to manage a cache here.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, AsyncIterator, Callable

import numpy as np

from . import voices as voices_mod
from .engine import DEFAULT_SAMPLE_RATE, build_qwen3_tts
from .pb import neural_tts_pb2 as pb
from .voices import LANG_TO_QWEN, VoiceEntry

log = logging.getLogger("neural_tts_provider_qwen3_tts.provider")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("ignoring %s=%r (not an int); using default %d", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("ignoring %s=%r (not a float); using default %.3f", name, raw, default)
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


class Qwen3TTSProvider:
    def __init__(self, eager_startup: bool = False) -> None:
        self._eager = eager_startup
        self._model: Any | None = None
        self._voices: dict[str, VoiceEntry] = {}
        self.sample_rate = DEFAULT_SAMPLE_RATE

    # ── voice management ───────────────────────────────────────────────

    def _rescan_voices(self) -> None:
        entries = voices_mod.scan_voices()
        self._voices = {e.voice_id: e for e in entries}

    def list_voices_pb(self) -> list[pb.Voice]:
        self._rescan_voices()
        return [voices_mod.to_pb(v) for v in self._voices.values()]

    # ── lifecycle ──────────────────────────────────────────────────────

    async def _ensure_runtime_loaded(self) -> None:
        if self._model is not None:
            return
        log.info("loading faster-qwen3-tts (this can take ~30 s on first run)")
        self._model = await asyncio.to_thread(build_qwen3_tts)
        try:
            sr = int(getattr(self._model, "sample_rate", 0))
            if sr:
                self.sample_rate = sr
        except (TypeError, ValueError):
            pass

    async def warmup(self) -> tuple[int, list[pb.Voice]]:
        self._rescan_voices()
        if not self._voices:
            log.warning(
                "no voices found — synthesise requests will fail until reference "
                "clips are dropped into ~/.local/share/neural-tts-daemon/voices/"
                "qwen3-tts/ and `neural-tts-ctl reload-voices` is invoked"
            )

        if self._eager:
            await self._ensure_runtime_loaded()
            if self._voices:
                first = next(iter(self._voices.values()))
                log.info("eager warmup: priming CUDA graphs with voice %s", first.voice_id)
                try:
                    await asyncio.to_thread(self._drain_warmup, first)
                except Exception:
                    log.exception("eager warmup synth failed (model loaded, continuing)")
            log.info("eager warmup complete; %d voice(s) available", len(self._voices))
        else:
            log.info(
                "lazy warmup: %d voice(s) enumerated, model load deferred to first synth",
                len(self._voices),
            )
        return self.sample_rate, self.list_voices_pb()

    async def shutdown(self) -> None:
        log.info("shutting down Qwen3-TTS provider")
        self._voices.clear()
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ── synthesis ──────────────────────────────────────────────────────

    async def synthesize_stream(
        self, *, voice: str, speed: float, lang: str, text: str
    ) -> AsyncIterator[np.ndarray]:
        await self._ensure_runtime_loaded()
        if voice not in self._voices:
            raise RuntimeError(
                f"unknown voice {voice!r}; available: "
                f"{sorted(self._voices) or 'NONE — drop clips into the voices/qwen3-tts dir'}"
            )
        entry = self._voices[voice]

        if abs(speed - 1.0) > 0.05:
            log.warning("Qwen3-TTS has no speed knob; ignoring speed=%.2f", speed)

        gen_kwargs = self._build_gen_kwargs(text)
        log.info(
            "synth voice=%s lang=%s chars=%d chunk_size=%d greedy=%s",
            voice, entry.lang, len(text),
            gen_kwargs.get("chunk_size"),
            not gen_kwargs.get("do_sample", True),
        )

        async for pcm in self._bridge_stream(entry, text, gen_kwargs):
            if pcm.size:
                yield pcm

    # ── streaming bridge: sync generator → async iterator ──────────────

    async def _bridge_stream(
        self,
        entry: VoiceEntry,
        text: str,
        gen_kwargs: dict,
    ) -> AsyncIterator[np.ndarray]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel: object = object()

        def producer() -> None:
            try:
                self._run_streaming(
                    entry, text, gen_kwargs,
                    emit=lambda pcm: loop.call_soon_threadsafe(queue.put_nowait, pcm),
                )
            except BaseException as e:  # noqa: BLE001 — re-raised on consumer
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
            try:
                await task
            except Exception:
                pass

    def _run_streaming(
        self,
        entry: VoiceEntry,
        text: str,
        gen_kwargs: dict,
        *,
        emit: Callable[[np.ndarray], None],
    ) -> None:
        """Drive FasterQwen3TTS.generate_voice_clone_streaming(...).

        Yields are `(np.ndarray float32, sample_rate int, timing dict)`. We
        track the model-reported sample rate; emit() forwards only PCM.
        """
        assert self._model is not None
        gen = self._model.generate_voice_clone_streaming(
            text=text,
            language=LANG_TO_QWEN[entry.lang],
            ref_audio=str(entry.wav_path),
            ref_text=entry.prompt_text,
            **gen_kwargs,
        )
        first_timing_logged = False
        for chunk, sr, timing in gen:
            if sr and sr != self.sample_rate:
                log.info("sample_rate %d → %d", self.sample_rate, sr)
                self.sample_rate = int(sr)
            if not first_timing_logged and isinstance(timing, dict):
                log.info("first-chunk timing: %s", timing)
                first_timing_logged = True
            arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if arr.size:
                emit(arr)

    def _drain_warmup(self, entry: VoiceEntry) -> None:
        """Eager warmup: capture CUDA graphs by running one tiny synth."""
        gen_kwargs = self._build_gen_kwargs("Warming up.")
        try:
            self._run_streaming(
                entry, "Warming up.", gen_kwargs,
                emit=lambda _pcm: None,
            )
        except Exception:
            log.exception("warmup streaming synth failed")

    def _build_gen_kwargs(self, text: str) -> dict:
        """Generation kwargs forwarded to generate_voice_clone_streaming.

        faster-qwen3-tts uses:
          chunk_size (default 12)   — codec-frame batching before each yield;
                                       lower = lower TTFA, slightly more overhead.
          max_new_tokens (2048)     — capped here to a per-char budget to keep
                                       short utterances from running long.
          do_sample/temperature/... — sampling controls; TTS_QWEN3_GREEDY=1
                                       switches to argmax (small steady win).
        """
        per_char = _env_int("TTS_QWEN3_MAX_NEW_TOKENS_PER_CHAR", 6)
        out: dict = {
            "max_new_tokens": min(2048, 32 + per_char * len(text)),
            "chunk_size": _env_int("TTS_QWEN3_CHUNK_SIZE", 12),
        }
        if _env_bool("TTS_QWEN3_GREEDY"):
            out["do_sample"] = False
        else:
            out["temperature"] = _env_float("TTS_QWEN3_TEMPERATURE", 0.9)
            out["top_k"] = _env_int("TTS_QWEN3_TOP_K", 50)
            out["top_p"] = _env_float("TTS_QWEN3_TOP_P", 1.0)
        out["repetition_penalty"] = _env_float("TTS_QWEN3_REPETITION_PENALTY", 1.05)
        return out
