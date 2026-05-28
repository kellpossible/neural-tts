"""Qwen3-TTS model load + device selection (faster-qwen3-tts backend).

Loads `Qwen/Qwen3-TTS-12Hz-0.6B-Base` onto CUDA via andimarafioti's
faster-qwen3-tts — a from-scratch CUDA-graph capture of the predictor +
talker that hits multi-x realtime on consumer GPUs without flash-attn,
vLLM, or Triton. Streams via the `generate_voice_clone_streaming(...)`
generator on `FasterQwen3TTS`.

Overridable via env:
  TTS_QWEN3_MODEL_PATH    explicit local model dir (else HF cache)
  TTS_QWEN3_DEVICE        e.g. "cuda", "cuda:1" (CPU is unsupported)
  TTS_QWEN3_DTYPE         bf16 / fp16 / fp32  (default bf16)
  TTS_QWEN3_ATTN          sdpa / flash_attention_2  (default sdpa)
  TTS_QWEN3_MAX_SEQ_LEN   int (default 2048)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("neural_tts_provider_qwen3_tts.engine")

HF_REPO_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
MODEL_DIR_NAME = "qwen3-tts-0.6b"
# Initial guess; provider overwrites from model.sample_rate post-load.
DEFAULT_SAMPLE_RATE = 24_000


def _data_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base) if base else Path.home() / ".local/share"
    return root / "neural-tts-daemon"


def resolve_model_dir() -> Path:
    override = os.environ.get("TTS_QWEN3_MODEL_PATH")
    if override:
        return Path(override)
    return _data_dir() / "models" / MODEL_DIR_NAME


def select_device() -> str:
    """Pick a CUDA device. faster-qwen3-tts requires CUDA."""
    import torch

    override = os.environ.get("TTS_QWEN3_DEVICE")
    if override:
        log.info("using TTS_QWEN3_DEVICE=%s", override)
        return override
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError(
        "faster-qwen3-tts requires CUDA — no compatible GPU detected. "
        "Switch providers (kokoro-onnx, moss-tts-nano) for CPU-only hosts."
    )


def _resolve_dtype():
    raw = os.environ.get("TTS_QWEN3_DTYPE", "").strip().lower()
    if not raw:
        return "bfloat16"
    table = {
        "bf16": "bfloat16", "bfloat16": "bfloat16",
        "fp16": "float16", "float16": "float16", "half": "float16",
        "fp32": "float32", "float32": "float32", "full": "float32",
    }
    if raw not in table:
        log.warning("ignoring TTS_QWEN3_DTYPE=%r (expected bf16/fp16/fp32)", raw)
        return "bfloat16"
    return table[raw]


def _resolve_attn_impl() -> str:
    raw = os.environ.get("TTS_QWEN3_ATTN", "").strip().lower()
    if raw in ("flash_attention_2", "sdpa"):
        return raw
    if raw:
        log.warning("ignoring TTS_QWEN3_ATTN=%r (expected sdpa/flash_attention_2)", raw)
    return "sdpa"


def build_qwen3_tts():
    """Load the model into FasterQwen3TTS. Returns the wrapper instance."""
    from faster_qwen3_tts import FasterQwen3TTS  # type: ignore[import-not-found]

    device = select_device()
    dtype = _resolve_dtype()
    attn = _resolve_attn_impl()
    try:
        max_seq_len = int(os.environ.get("TTS_QWEN3_MAX_SEQ_LEN", "2048"))
    except ValueError:
        max_seq_len = 2048

    model_dir = resolve_model_dir()
    source = str(model_dir) if model_dir.exists() else HF_REPO_ID

    log.info(
        "loading faster-qwen3-tts from %s onto %s (dtype=%s, attn=%s, max_seq_len=%d)",
        source, device, dtype, attn, max_seq_len,
    )
    try:
        model = FasterQwen3TTS.from_pretrained(
            source,
            device=device,
            dtype=dtype,
            attn_implementation=attn,
            max_seq_len=max_seq_len,
        )
    except (ImportError, RuntimeError, ValueError) as e:
        if attn == "flash_attention_2":
            log.warning("flash_attention_2 init failed (%s); retrying with sdpa", e)
            model = FasterQwen3TTS.from_pretrained(
                source,
                device=device,
                dtype=dtype,
                attn_implementation="sdpa",
                max_seq_len=max_seq_len,
            )
        else:
            raise

    log.info("faster-qwen3-tts ready (sample_rate=%s)", getattr(model, "sample_rate", "?"))
    return model
