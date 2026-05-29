"""Qwen3-TTS provider (faster-qwen3-tts backend).

Streaming: drives `FasterQwen3TTS.generate_voice_clone_streaming(...)`
and forwards each yielded PCM chunk to the daemon. The wrapper streams
natively from the codec, so we hand it the whole utterance — no text
chunker by default.

Voice-prompt caching: faster-qwen3-tts keeps an internal cache keyed
by (ref_audio, ref_text). Passing the same paths/text reuses the
precomputed ref_code + speaker embedding.

Optional jitter-buffered mode (TTS_QWEN3_CHUNKER=1): split the input
via .chunker, then for each piece run an adaptive-prebuffer bridge
that holds back PCM until enough audio is accumulated to cover the
estimated synth deficit (D*(1-RTF)/RTF + safety). Trades TTFA for
gap-free playback when sustained RTF < 1.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

import numpy as np

from . import chunker
from . import voices as voices_mod
from .engine import DEFAULT_SAMPLE_RATE, build_qwen3_tts
from .pb import neural_tts_pb2 as pb
from .voices import LANG_TO_QWEN, VoiceEntry

log = logging.getLogger("neural_tts_provider_qwen3_tts.provider")

# faster-qwen3-tts emits one codec step at 12 Hz per step in the timing dict.
CODEC_HZ = 12


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


def _chunker_enabled() -> bool:
    return _env_bool("TTS_QWEN3_CHUNKER", False)


@dataclass
class _StreamStats:
    """Rolling RTF + chars/sec across pieces of a single request.

    Persists across chunks so that estimates improve after the first
    piece is calibrated.
    """
    audio_s: float = 0.0
    wall_s: float = 0.0
    chars: int = 0

    def update_from_timing(self, timing: dict, chunk_audio_s: float) -> None:
        """Accumulate one yield from the model."""
        self.audio_s += chunk_audio_s
        ms = float(timing.get("decode_ms") or 0.0) + float(timing.get("prefill_ms") or 0.0)
        self.wall_s += ms / 1000.0

    def add_chars(self, n: int) -> None:
        self.chars += n

    @property
    def rtf(self) -> float | None:
        if self.wall_s <= 0:
            return None
        return self.audio_s / self.wall_s

    def chars_per_sec(self, bootstrap: float) -> float:
        if self.audio_s <= 0 or self.chars <= 0:
            return bootstrap
        return self.chars / self.audio_s


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
        use_chunker = _chunker_enabled()
        log.info(
            "synth voice=%s lang=%s chars=%d chunk_size=%d greedy=%s chunker=%s",
            voice, entry.lang, len(text),
            gen_kwargs.get("chunk_size"),
            not gen_kwargs.get("do_sample", True),
            "on" if use_chunker else "off",
        )

        if not use_chunker:
            # Fast path — unchanged behaviour, raw passthrough.
            async for pcm in self._bridge_stream(entry, text, gen_kwargs):
                if pcm.size:
                    yield pcm
            return

        # Buffered path — chunker + per-piece adaptive jitter buffer.
        stats = _StreamStats()
        pieces = list(chunker.chunk(text, entry.lang))
        log.info("chunker produced %d piece(s) (target=%d, hard_cap=%d)",
                 len(pieces), chunker.TARGET_CHARS, chunker.HARD_CAP_CHARS)
        request_start = time.monotonic()
        for i, piece in enumerate(pieces):
            async for pcm in self._bridge_stream_buffered(entry, piece, gen_kwargs, stats, i):
                if pcm.size:
                    yield pcm
        request_wall = time.monotonic() - request_start
        log.info(
            "synth done: pieces=%d total_audio=%.2fs synth_wall=%.2fs "
            "request_wall=%.2fs rolling_rtf=%s chars_per_sec=%.1f",
            len(pieces), stats.audio_s, stats.wall_s, request_wall,
            f"{stats.rtf:.2f}" if stats.rtf is not None else "n/a",
            stats.chars_per_sec(_env_float("TTS_QWEN3_CHARS_PER_SEC_BOOTSTRAP", 15.0)),
        )

    # ── streaming bridge: sync generator → async iterator ──────────────

    async def _bridge_stream(
        self,
        entry: VoiceEntry,
        text: str,
        gen_kwargs: dict,
    ) -> AsyncIterator[np.ndarray]:
        """Fast-path bridge — emit PCM as soon as the model yields it."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel: object = object()
        cancel = threading.Event()

        def producer() -> None:
            try:
                self._run_streaming(
                    entry, text, gen_kwargs,
                    emit=lambda pcm, _timing: loop.call_soon_threadsafe(
                        queue.put_nowait, pcm
                    ),
                    cancel=cancel,
                )
            except BaseException as e:  # noqa: BLE001
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
            cancel.set()
            try:
                await task
            except Exception:
                pass

    async def _bridge_stream_buffered(
        self,
        entry: VoiceEntry,
        piece: str,
        gen_kwargs: dict,
        stats: _StreamStats,
        piece_index: int,
    ) -> AsyncIterator[np.ndarray]:
        """Adaptive-prebuffer bridge for one chunker piece.

        Holds PCM in a deque until either (a) we've accumulated enough
        audio to cover the expected synth deficit, or (b) the producer
        signals end-of-piece. Once unblocked, switches to passthrough
        for the rest of the piece.
        """
        bootstrap_cps = _env_float("TTS_QWEN3_CHARS_PER_SEC_BOOTSTRAP", 15.0)
        safety_s = _env_int("TTS_QWEN3_JITTER_SAFETY_MS", 200) / 1000.0
        initial_s = _env_int("TTS_QWEN3_JITTER_INITIAL_MS", 500) / 1000.0

        est_D = max(1.0, len(piece) / stats.chars_per_sec(bootstrap_cps))
        rtf = stats.rtf
        if rtf is None:
            target_s = initial_s
            target_src = "initial"
        elif rtf >= 1.0:
            target_s = 0.0
            target_src = "fast-path"
        else:
            target_s = est_D * (1.0 - rtf) / rtf + safety_s
            target_src = f"rtf={rtf:.2f}"
        log.info(
            "piece=%d chars=%d est_D=%.2fs target_prebuffer=%.2fs (%s)",
            piece_index, len(piece), est_D, target_s, target_src,
        )

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel: object = object()
        cancel = threading.Event()

        def producer() -> None:
            try:
                self._run_streaming(
                    entry, piece, gen_kwargs,
                    emit=lambda pcm, timing: loop.call_soon_threadsafe(
                        queue.put_nowait, (pcm, timing)
                    ),
                    cancel=cancel,
                )
            except BaseException as e:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, e)
            else:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        task = asyncio.create_task(asyncio.to_thread(producer))
        stats.add_chars(len(piece))

        buf: deque[np.ndarray] = deque()
        buf_s = 0.0
        passthrough = (target_s <= 0.0)

        piece_start = time.monotonic()
        prebuffer_fill_wall_s: float | None = None
        piece_audio_s = 0.0
        chunks_emitted = 0
        # Track the snapshot of stats at piece start so we can derive a
        # per-piece RTF (the rolling stats include all earlier pieces).
        piece_stats_audio_start = stats.audio_s
        piece_stats_wall_start = stats.wall_s

        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    # End-of-piece flush. Anything still in buf means the
                    # producer finished before the prebuffer target was hit —
                    # i.e. piece was shorter than estimated, so we just emit
                    # whatever we accumulated.
                    while buf:
                        yield buf.popleft()
                        chunks_emitted += 1
                    # Per-piece summary
                    piece_wall = time.monotonic() - piece_start
                    piece_audio_synth = stats.audio_s - piece_stats_audio_start
                    piece_synth_wall = stats.wall_s - piece_stats_wall_start
                    piece_rtf = (
                        piece_audio_synth / piece_synth_wall
                        if piece_synth_wall > 0 else float("nan")
                    )
                    log.info(
                        "piece=%d done: audio=%.2fs synth_wall=%.2fs piece_rtf=%.2f "
                        "wall_elapsed=%.2fs prebuffer_fill=%s chunks_out=%d",
                        piece_index, piece_audio_synth, piece_synth_wall, piece_rtf,
                        piece_wall,
                        f"{prebuffer_fill_wall_s:.2f}s" if prebuffer_fill_wall_s is not None else "none",
                        chunks_emitted,
                    )
                    return
                if isinstance(item, BaseException):
                    raise item
                pcm, timing = item  # type: ignore[misc]
                chunk_audio_s = pcm.size / float(self.sample_rate)
                stats.update_from_timing(timing, chunk_audio_s)
                piece_audio_s += chunk_audio_s
                if passthrough:
                    yield pcm
                    chunks_emitted += 1
                    continue
                buf.append(pcm)
                buf_s += chunk_audio_s
                if buf_s >= target_s:
                    prebuffer_fill_wall_s = time.monotonic() - piece_start
                    log.info(
                        "piece=%d prebuffer filled: audio=%.2fs (target %.2fs) "
                        "after wall=%.2fs rolling_rtf=%s; switching to passthrough",
                        piece_index, buf_s, target_s, prebuffer_fill_wall_s,
                        f"{stats.rtf:.2f}" if stats.rtf is not None else "n/a",
                    )
                    passthrough = True
                    while buf:
                        yield buf.popleft()
                        chunks_emitted += 1
        finally:
            cancel.set()
            try:
                await task
            except Exception:
                pass

    # ── synchronous helpers (run inside a worker thread) ───────────────

    def _run_streaming(
        self,
        entry: VoiceEntry,
        text: str,
        gen_kwargs: dict,
        *,
        emit: Callable[[np.ndarray, dict], None],
        cancel: threading.Event | None = None,
    ) -> None:
        """Drive FasterQwen3TTS.generate_voice_clone_streaming(...).

        Yields are `(pcm, sample_rate, timing dict)`. emit() receives
        (pcm, timing); the consumer is responsible for forwarding only
        PCM downstream.
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
            if cancel is not None and cancel.is_set():
                log.info("cancel event received; aborting model generator")
                try:
                    gen.close()
                except Exception:
                    pass
                return
            if sr and sr != self.sample_rate:
                log.info("sample_rate %d → %d", self.sample_rate, sr)
                self.sample_rate = int(sr)
            if not first_timing_logged and isinstance(timing, dict):
                log.info("first-chunk timing: %s", timing)
                first_timing_logged = True
            arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if arr.size:
                emit(arr, timing if isinstance(timing, dict) else {})

    def _drain_warmup(self, entry: VoiceEntry) -> None:
        """Eager warmup: capture CUDA graphs by running one tiny synth."""
        gen_kwargs = self._build_gen_kwargs("Warming up.")
        try:
            self._run_streaming(
                entry, "Warming up.", gen_kwargs,
                emit=lambda _pcm, _timing: None,
            )
        except Exception:
            log.exception("warmup streaming synth failed")

    def _build_gen_kwargs(self, text: str) -> dict:
        per_char = _env_int("TTS_QWEN3_MAX_NEW_TOKENS_PER_CHAR", 6)
        out: dict = {
            "max_new_tokens": min(2048, 32 + per_char * len(text)),
            "chunk_size": _env_int("TTS_QWEN3_CHUNK_SIZE", 12),
        }
        # TTS_QWEN3_GREEDY=1 switches from sampled decoding (default) to
        # greedy decoding: always pick the highest-probability next token
        # instead of sampling from the temperature/top-k/top-p distribution.
        # Modest speedup (skips the sampler step per token), fully
        # deterministic output, slightly less prosodic variation. Usually a
        # win for voice-clone TTS because the reference clip already pins
        # down voice character; drop it if synthesis sounds robotic.
        if _env_bool("TTS_QWEN3_GREEDY"):
            out["do_sample"] = False
        else:
            out["temperature"] = _env_float("TTS_QWEN3_TEMPERATURE", 0.9)
            out["top_k"] = _env_int("TTS_QWEN3_TOP_K", 50)
            out["top_p"] = _env_float("TTS_QWEN3_TOP_P", 1.0)
        out["repetition_penalty"] = _env_float("TTS_QWEN3_REPETITION_PENALTY", 1.05)
        return out
