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


def _cache_dir() -> Path:
    """Writable cache root. Unlike _data_dir(), this stays writable under the
    daemon's systemd sandbox (ReadWritePaths=%h/.cache/neural-tts-daemon)."""
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
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


def _resolve_flashinfer_mode() -> str:
    """Parse TTS_OMNIVOICE_FLASHINFER. Returns one of:

      "auto"        →  unset/"auto": apply when on CUDA and flashinfer imports
      "off"         →  "0" / "false" / "no" / "off": never apply
      "on"          →  "1" / "true" / "yes" / "on": apply, warn loudly if unavailable
      "cuda-graph"  →  like "on", plus enable_cuda_graph=True (per-shape CUDA
                       graphs; upstream recommends for batch=1, but graph
                       capture adds startup cost per distinct input shape)
    """
    raw = os.environ.get("TTS_OMNIVOICE_FLASHINFER", "").strip().lower()
    if not raw or raw == "auto":
        return "auto"
    if raw in {"0", "false", "no", "off"}:
        return "off"
    if raw in {"1", "true", "yes", "on"}:
        return "on"
    if raw in {"cuda-graph", "cuda_graph", "cudagraph"}:
        return "cuda-graph"
    log.warning("unrecognised TTS_OMNIVOICE_FLASHINFER=%r; treating as 'auto'", raw)
    return "auto"


def _maybe_apply_flashinfer(model, device: str) -> bool:
    """Patch the model with upstream's FlashInfer decode path if possible.

    Returns True if the patch was applied. FlashInfer is CUDA-only; on other
    devices this is a no-op (a warning is logged if the user explicitly
    requested it).
    """
    mode = _resolve_flashinfer_mode()
    if mode == "off":
        return False
    explicit = mode in {"on", "cuda-graph"}
    if not device.startswith("cuda"):
        if explicit:
            log.warning(
                "TTS_OMNIVOICE_FLASHINFER=%s requested but device is %s — "
                "flashinfer is CUDA-only, skipping", mode, device,
            )
        return False
    # flashinfer writes its JIT log/cubins under $FLASHINFER_WORKSPACE_BASE/
    # .cache/flashinfer at import time; the default (~) is read-only under the
    # daemon's systemd sandbox, so point it at our writable cache dir.
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(_cache_dir()))
    try:
        from omnivoice.models.omnivoice_flashinfer import (  # type: ignore[import-not-found]
            apply_flashinfer,
        )
    except ImportError as e:
        msg = (
            "flashinfer not available (%s) — install the provider with "
            "`--extra flashinfer` for ~2-2.9x faster decoding on CUDA"
        )
        (log.warning if explicit else log.info)(msg, e)
        return False
    try:
        apply_flashinfer(model, enable_cuda_graph=(mode == "cuda-graph"))
    except Exception:
        log.exception("apply_flashinfer() failed; continuing with the eager model")
        return False
    log.info(
        "FlashInfer decode path enabled (cuda_graph=%s) — expect ~2-2.9x "
        "faster per-chunk synthesis", mode == "cuda-graph",
    )
    return True


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

    flashinfer_applied = _maybe_apply_flashinfer(model, device)

    compile_mode = _resolve_compile_mode()
    if compile_mode and flashinfer_applied:
        # FlashInfer replaces the decode loop with hand-fused kernels and
        # per-generation attention plans; torch.compile on top is untested
        # upstream and would at best re-trace what's already fused.
        log.warning(
            "TTS_OMNIVOICE_COMPILE=%r ignored because FlashInfer is active — "
            "set TTS_OMNIVOICE_FLASHINFER=off to use torch.compile instead",
            compile_mode,
        )
        compile_mode = None
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

    log.info(
        "OmniVoice ready (sample_rate=%d, flashinfer=%s, compile=%s)",
        SAMPLE_RATE, "on" if flashinfer_applied else "off", compile_mode or "off",
    )
    return model
