"""OmniVoice model load + device selection.

Loads `k2-fsa/OmniVoice` (https://github.com/k2-fsa/OmniVoice) onto the best
available accelerator (CUDA → XPU → MPS → CPU). The `omnivoice` Python package
is pulled in via pyproject.toml as a pinned git dependency, so no sys.path
manipulation is needed here.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("neural_tts_provider_omnivoice.engine")

HF_REPO_ID = "k2-fsa/OmniVoice"
MODEL_DIR_NAME = "omnivoice"
SAMPLE_RATE = 24_000


def _data_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base) if base else Path.home() / ".local/share"
    return root / "neural-tts-daemon"


def resolve_model_dir() -> Path:
    """Local directory containing the snapshot_download'd model files."""
    override = os.environ.get("TTS_OMNIVOICE_MODEL_PATH")
    if override:
        return Path(override)
    return _data_dir() / "models" / MODEL_DIR_NAME


def select_device() -> str:
    """Pick the best accelerator. Honour TTS_OMNIVOICE_DEVICE if set."""
    import torch

    override = os.environ.get("TTS_OMNIVOICE_DEVICE")
    if override:
        log.info("using TTS_OMNIVOICE_DEVICE=%s", override)
        return override
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    log.warning(
        "no GPU detected; falling back to CPU — synthesis will be well below "
        "realtime. Pick a smaller provider (kokoro-onnx, moss-tts-nano) for "
        "interactive use, or set TTS_OMNIVOICE_DEVICE to override."
    )
    return "cpu"


def _resolve_compile_mode() -> str | None:
    """Parse TTS_OMNIVOICE_COMPILE. Returns torch.compile mode or None if off.

    Accepted values (case-insensitive):
      "" / unset / "0" / "false" / "no" / "off"  →  None (eager, default)
      "1" / "true" / "yes" / "on"                →  "default"
      "default" / "reduce-overhead" / "max-autotune"  →  passed through verbatim

    "reduce-overhead" uses CUDA graphs — fastest steady-state, but it pins
    shapes so dynamic text lengths can break it; if you hit recompilation
    storms or shape errors, fall back to "default".
    """
    raw = os.environ.get("TTS_OMNIVOICE_COMPILE", "").strip().lower()
    if not raw or raw in {"0", "false", "no", "off"}:
        return None
    if raw in {"1", "true", "yes", "on"}:
        return "default"
    return raw


def build_omnivoice():
    """Load the OmniVoice model onto the best available device.

    Returns the model instance. The caller drives it via model.generate(...).
    """
    import torch
    from omnivoice import OmniVoice  # type: ignore[import-not-found]

    device = select_device()
    # fp16 on real GPUs; fp32 on CPU (fp16 on CPU is dog-slow without AMX).
    dtype = torch.float16 if device != "cpu" else torch.float32

    model_dir = resolve_model_dir()
    # Prefer the local snapshot if present; otherwise let HF resolve from cache.
    source = str(model_dir) if model_dir.exists() else HF_REPO_ID

    log.info("loading OmniVoice from %s onto %s (dtype=%s)", source, device, dtype)
    model = OmniVoice.from_pretrained(source, device_map=device, dtype=dtype)

    compile_mode = _resolve_compile_mode()
    if compile_mode:
        log.info(
            "wrapping model in torch.compile(mode=%r) — first synth call will "
            "pay JIT compile cost (often 30-60 s); steady-state should be "
            "~20-40%% faster per diffusion step. Set TTS_OMNIVOICE_COMPILE= "
            "(empty) to disable if you hit recompile storms or shape errors.",
            compile_mode,
        )
        try:
            model = torch.compile(model, mode=compile_mode)
        except Exception:
            log.exception("torch.compile() raised at wrap time; using eager model instead")

    log.info("OmniVoice ready (sample_rate=%d, compile=%s)", SAMPLE_RATE, compile_mode or "off")
    return model
