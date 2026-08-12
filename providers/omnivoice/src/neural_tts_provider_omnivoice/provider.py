"""OmniVoice provider.

Implements warmup / list_voices / synthesize_stream / shutdown against the
neural-tts-daemon provider protocol. Voices are user-supplied reference
clips (see voices.py). Each synthesize request is text-chunked
(see chunker.py) and emitted as a series of per-chunk PCM blobs.

OmniVoice's public Python API (`model.generate`) is one-shot per call — it
does not expose an intra-utterance streaming iterator. We approximate
streaming at the sentence/clause level via chunker.py: split the input,
synth each chunk, yield its PCM, repeat. This gives reasonable
time-to-first-audio for paragraph-length input.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from pathlib import Path
from typing import AsyncIterator

import numpy as np

from . import voices as voices_mod
from .chunker import chunk as chunk_text
from .engine import SAMPLE_RATE, _cache_dir, build_omnivoice
from .pb import neural_tts_pb2 as pb
from .voices import VoiceEntry

log = logging.getLogger("neural_tts_provider_omnivoice.provider")

# Default diffusion-step count. Upstream's default is 32; 16 is the
# faster-inference value the README endorses. Overridable per-deployment via
# the TTS_OMNIVOICE_NUM_STEP env var — it's a direct quality/latency knob,
# higher = better fidelity, lower = lower TTFA + per-chunk synth time.
_DEFAULT_NUM_STEP = 16


def _resolve_num_step() -> int:
    raw = os.environ.get("TTS_OMNIVOICE_NUM_STEP")
    if not raw:
        return _DEFAULT_NUM_STEP
    try:
        v = int(raw)
    except ValueError:
        log.warning("TTS_OMNIVOICE_NUM_STEP=%r is not an int; using default %d",
                    raw, _DEFAULT_NUM_STEP)
        return _DEFAULT_NUM_STEP
    if v < 1:
        log.warning("TTS_OMNIVOICE_NUM_STEP=%d must be >= 1; using default %d",
                    v, _DEFAULT_NUM_STEP)
        return _DEFAULT_NUM_STEP
    return v


NUM_STEP = _resolve_num_step()

# Upstream generate() pads 0.1 s of silence per side and applies a 0.1 s
# fade-in/out to every clip. Each streamed chunk is its own generate() call,
# so back-to-back chunks stack ~200 ms of dead air at every join; we default
# to half that. 0 disables either entirely.
_DEFAULT_PAD_DURATION = 0.05
_DEFAULT_FADE_DURATION = 0.05


def _resolve_duration(env_name: str, default: float) -> float:
    raw = os.environ.get(env_name)
    if not raw:
        return default
    try:
        v = float(raw)
    except ValueError:
        log.warning("%s=%r is not a number; using default %s", env_name, raw, default)
        return default
    if v < 0:
        log.warning("%s=%s must be >= 0; using default %s", env_name, v, default)
        return default
    return v


PAD_DURATION = _resolve_duration("TTS_OMNIVOICE_PAD_DURATION", _DEFAULT_PAD_DURATION)
FADE_DURATION = _resolve_duration("TTS_OMNIVOICE_FADE_DURATION", _DEFAULT_FADE_DURATION)


def _prompt_cache_path(entry: VoiceEntry) -> Path | None:
    """Disk location for this voice's serialized VoiceClonePrompt.

    The filename embeds a digest of the reference clip's (path, mtime, size)
    and the transcript, so editing either invalidates the cache naturally —
    the old file is just never looked up again (and is swept on next save).
    """
    try:
        st = entry.wav_path.stat()
    except OSError:
        return None
    h = hashlib.sha1()
    h.update(str(entry.wav_path).encode())
    h.update(f":{st.st_mtime_ns}:{st.st_size}:".encode())
    h.update((entry.prompt_text or "").encode())
    # Lives under the cache dir (not the data dir): prompts are rebuildable,
    # and the daemon's systemd sandbox only grants writes to the cache dir.
    return (
        _cache_dir() / "prompts" / "omnivoice"
        / f"{entry.voice_id}.{entry.lang}-{h.hexdigest()[:16]}.pt"
    )


class OmnivoiceProvider:
    def __init__(self, eager_startup: bool = False) -> None:
        self._eager = eager_startup
        self._model = None
        self._voices: dict[str, VoiceEntry] = {}
        # voice_id → cached VoiceClonePrompt from model.create_voice_clone_prompt().
        # OmniVoice rebuilds prompt audio tokens from the raw WAV every call by
        # default — caching the prompt object skips the librosa-load + audio
        # encoder work on every synth (hundreds of ms steady-state per chunk).
        self._prompt_cache: dict[str, object] = {}
        self.sample_rate = SAMPLE_RATE
        log.info(
            "diffusion num_step=%d (default=%d, env=%r)",
            NUM_STEP, _DEFAULT_NUM_STEP, os.environ.get("TTS_OMNIVOICE_NUM_STEP"),
        )

    # ── voice management ───────────────────────────────────────────────

    def _rescan_voices(self) -> None:
        entries = voices_mod.scan_voices()
        self._voices = {e.voice_id: e for e in entries}
        # Drop cache entries whose source disappeared.
        for stale in [k for k in self._prompt_cache if k not in self._voices]:
            self._prompt_cache.pop(stale, None)

    def list_voices_pb(self) -> list[pb.Voice]:
        self._rescan_voices()
        return [voices_mod.to_pb(v) for v in self._voices.values()]

    # ── lifecycle ──────────────────────────────────────────────────────

    async def _ensure_runtime_loaded(self) -> None:
        if self._model is not None:
            return
        log.info("loading OmniVoice (this can take ~30 s on first run)")
        self._model = await asyncio.to_thread(build_omnivoice)

    async def warmup(self) -> tuple[int, list[pb.Voice]]:
        """Quick voice scan + sample-rate enumeration. Model load is deferred to
        the first `synthesize_stream` call unless `eager_startup=True`."""
        self._rescan_voices()
        if not self._voices:
            log.warning(
                "no voices found — synthesise requests will fail until reference clips "
                "are dropped into ~/.local/share/neural-tts-daemon/voices/omnivoice/ "
                "and `neural-tts-ctl reload-voices` is invoked"
            )

        if self._eager:
            await self._ensure_runtime_loaded()
            log.info("eager warmup complete; %d voice(s) available", len(self._voices))
        else:
            log.info(
                "lazy warmup: %d voice(s) enumerated, model load deferred to first synth",
                len(self._voices),
            )
        return SAMPLE_RATE, self.list_voices_pb()

    async def shutdown(self) -> None:
        log.info("shutting down OmniVoice provider")
        self._voices.clear()
        self._prompt_cache.clear()
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
                f"{sorted(self._voices) or 'NONE — drop clips into the voices/omnivoice dir'}"
            )
        entry = self._voices[voice]
        # OmniVoice infers the target language from the reference clip + text;
        # the request `lang` arg is informational. We use the voice's own
        # `lang` to pick a sane pysbd segmenter.
        synth_lang = entry.lang

        chunks = list(chunk_text(text, synth_lang))
        if not chunks:
            return
        log.info(
            "synth voice=%s lang=%s chunks=%d total_chars=%d",
            voice, synth_lang, len(chunks), sum(len(c) for c in chunks),
        )

        for i, c in enumerate(chunks):
            log.info("  chunk %d/%d (%d chars): %r", i + 1, len(chunks), len(c), c[:60])
            samples = await asyncio.to_thread(self._synth_one, entry, c, speed)
            if samples.size:
                yield samples

    # ── one-shot helper that runs inside a thread ──────────────────────

    def _synth_one(self, entry: VoiceEntry, text: str, speed: float) -> np.ndarray:
        """Single forward pass returning float32 PCM at SAMPLE_RATE.

        Runs synchronously in a worker thread. The reference clip's prompt
        (audio tokens + transcript) is built once and cached per voice via
        `OmniVoice.create_voice_clone_prompt`; subsequent synths skip the
        librosa-load + audio-encoder pass.
        """
        assert self._model is not None  # _ensure_runtime_loaded ran already

        prompt = self._get_clone_prompt(entry)

        gen_kwargs: dict = {
            "text": text,
            "voice_clone_prompt": prompt,
            "num_step": NUM_STEP,
            # Upstream README: "Performance is slightly better if you specify
            # the language." Pass the voice's BCP-47 primary subtag.
            "language": entry.lang,
            "pad_duration": PAD_DURATION,
            "fade_duration": FADE_DURATION,
        }
        if abs(speed - 1.0) > 0.01:
            gen_kwargs["speed"] = float(speed)

        out = self._model.generate(**gen_kwargs)

        if isinstance(out, list):
            arrays = [np.asarray(x, dtype=np.float32).reshape(-1) for x in out if x is not None]
            if not arrays:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)

    def _get_clone_prompt(self, entry: VoiceEntry):
        """Build (and cache) the reusable VoiceClonePrompt for a voice.

        Upstream's `create_voice_clone_prompt` accepts the reference audio
        path/tensor and an optional transcript; without a transcript it runs
        Whisper to auto-transcribe. The returned prompt holds audio tokens
        ready for the diffusion forward pass, so reusing it skips both
        per-call audio loading and audio-tokenizer encoding.

        Prompts are also persisted to disk (VoiceClonePrompt.save/load), so
        the librosa + audio-encoder + Whisper work is paid once per reference
        clip, not once per daemon lifetime.
        """
        cached = self._prompt_cache.get(entry.voice_id)
        if cached is not None:
            return cached

        from omnivoice import VoiceClonePrompt  # type: ignore[import-not-found]

        disk_path = _prompt_cache_path(entry)
        if disk_path is not None and disk_path.exists():
            try:
                prompt = VoiceClonePrompt.load(str(disk_path))
            except Exception as e:
                log.warning("discarding unreadable prompt cache %s: %s", disk_path, e)
                disk_path.unlink(missing_ok=True)
            else:
                log.info("loaded voice clone prompt for %s from cache %s",
                         entry.voice_id, disk_path)
                self._prompt_cache[entry.voice_id] = prompt
                return prompt

        log.info("building voice clone prompt for %s from %s",
                 entry.voice_id, entry.wav_path)
        kwargs: dict = {"ref_audio": str(entry.wav_path)}
        if entry.prompt_text:
            kwargs["ref_text"] = entry.prompt_text
        prompt = self._model.create_voice_clone_prompt(**kwargs)
        self._prompt_cache[entry.voice_id] = prompt

        if disk_path is not None:
            try:
                disk_path.parent.mkdir(parents=True, exist_ok=True)
                # Sweep caches keyed to older versions of this voice's clip.
                for stale in disk_path.parent.glob(f"{entry.voice_id}.{entry.lang}-*.pt"):
                    if stale != disk_path:
                        stale.unlink(missing_ok=True)
                prompt.save(str(disk_path))
                log.info("saved voice clone prompt cache to %s", disk_path)
            except Exception as e:
                log.warning("could not persist prompt cache to %s: %s", disk_path, e)
        return prompt
