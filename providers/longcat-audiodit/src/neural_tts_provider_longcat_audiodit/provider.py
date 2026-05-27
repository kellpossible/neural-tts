"""LongCat-AudioDiT provider.

Implements warmup / list_voices / synthesize_stream / shutdown against the
neural-tts-daemon provider protocol. Voices are user-supplied reference
clips (see voices.py). Each synthesize request is text-chunked
(see chunker.py) and emitted as a series of per-chunk PCM blobs.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import numpy as np

from . import voices as voices_mod
from .chunker import chunk as chunk_text
from .engine import SAMPLE_RATE, build_longcat
from .pb import neural_tts_pb2 as pb
from .voices import VoiceEntry

log = logging.getLogger("neural_tts_provider_longcat_audiodit.provider")

# Diffusion hyperparams (defaults from upstream README).
DEFAULT_STEPS = 16
DEFAULT_CFG = 4.0
DEFAULT_GUIDANCE = "apg"  # recommended for voice cloning


class LongCatProvider:
    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._voices: dict[str, VoiceEntry] = {}
        self._prompt_cache: dict[str, "object"] = {}  # voice_id -> cached prompt wav tensor
        self.sample_rate = SAMPLE_RATE

    # ── voice management ───────────────────────────────────────────────

    def _rescan_voices(self) -> None:
        entries = voices_mod.scan_voices()
        self._voices = {e.voice_id: e for e in entries}
        # Drop any prompt cache entries whose source disappeared.
        for stale in [k for k in self._prompt_cache if k not in self._voices]:
            self._prompt_cache.pop(stale, None)

    def list_voices_pb(self) -> list[pb.Voice]:
        self._rescan_voices()
        return [voices_mod.to_pb(v) for v in self._voices.values()]

    # ── lifecycle ──────────────────────────────────────────────────────

    async def warmup(self) -> tuple[int, list[pb.Voice]]:
        log.info("loading LongCat-AudioDiT (this may take ~30 s)")
        # Heavy: torch + transformers + AudioDiT. Run in a thread so the
        # event loop stays responsive while the daemon waits.
        self._model, self._tokenizer = await asyncio.to_thread(build_longcat)
        self._rescan_voices()

        if self._voices:
            first = next(iter(self._voices.values()))
            log.info("warming up CUDA graphs with voice %s", first.voice_id)
            try:
                await asyncio.to_thread(
                    self._synth_one, first, first.lang, "Warming up."
                )
            except Exception:
                log.exception("warmup synth failed (model loaded, continuing anyway)")
        else:
            log.warning(
                "no voices found — synthesise requests will fail until reference clips "
                "are dropped into ~/.local/share/neural-tts-daemon/voices/longcat/ "
                "and `neural-tts-ctl reload-voices` is invoked"
            )

        log.info("warmup complete; %d voice(s) available", len(self._voices))
        return SAMPLE_RATE, self.list_voices_pb()

    async def shutdown(self) -> None:
        log.info("shutting down LongCat provider")
        self._voices.clear()
        self._prompt_cache.clear()
        self._tokenizer = None
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
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("provider not warmed up")
        if voice not in self._voices:
            raise RuntimeError(
                f"unknown voice {voice!r}; available: {sorted(self._voices) or 'NONE — drop clips into the voices/longcat dir'}"
            )
        entry = self._voices[voice]

        # LongCat lang is encoded in the voice itself (the reference clip was
        # recorded in that language); the request `lang` arg is informational.
        synth_lang = entry.lang

        if abs(speed - 1.0) > 0.05:
            log.warning(
                "LongCat does not support a speed knob; ignoring speed=%.2f", speed
            )

        chunks = list(chunk_text(text, synth_lang))
        if not chunks:
            return
        log.info(
            "synth voice=%s lang=%s chunks=%d total_chars=%d",
            voice, synth_lang, len(chunks), sum(len(c) for c in chunks),
        )

        for i, chunk in enumerate(chunks):
            log.info("  chunk %d/%d (%d chars): %r", i + 1, len(chunks), len(chunk), chunk[:60])
            samples = await asyncio.to_thread(self._synth_one, entry, synth_lang, chunk)
            if samples.size:
                yield samples

    # ── one-shot helper that runs inside a thread (CUDA call) ──────────

    def _synth_one(self, entry: VoiceEntry, lang: str, text: str) -> np.ndarray:
        """Single forward pass returning float32 PCM at SAMPLE_RATE.

        Runs synchronously in a worker thread. Loads (and caches) the prompt
        audio for `entry`, tokenises `text`, and invokes the model.
        """
        import torch

        prompt_audio = self._get_prompt_audio(entry)

        inputs = self._tokenizer(
            [text],
            padding="longest",
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")

        with torch.inference_mode():
            output = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_audio=prompt_audio,
                prompt_text=entry.prompt_text,
                steps=DEFAULT_STEPS,
                cfg_strength=DEFAULT_CFG,
                guidance_method=DEFAULT_GUIDANCE,
            )

        # `output.waveform` shape is (batch, channels?, samples) — squeeze to 1D.
        wav = output.waveform.detach().to("cpu", dtype=torch.float32).numpy()
        wav = np.asarray(wav).reshape(-1)
        return wav

    def _get_prompt_audio(self, entry: VoiceEntry):
        """Load + cache the prompt waveform for a voice.

        LongCat expects the prompt audio as a float tensor at the model's
        target sample rate. We use librosa to load + resample to 24 kHz mono
        and keep the tensor cached on-device for the life of the process.
        """
        cached = self._prompt_cache.get(entry.voice_id)
        if cached is not None:
            return cached

        import librosa
        import torch

        log.info("loading prompt audio for voice %s from %s", entry.voice_id, entry.wav_path)
        wav, _sr = librosa.load(str(entry.wav_path), sr=SAMPLE_RATE, mono=True)
        tensor = torch.from_numpy(wav).to("cuda", dtype=torch.float16).unsqueeze(0)
        self._prompt_cache[entry.voice_id] = tensor
        return tensor
