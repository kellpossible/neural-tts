"""Engine setup: ONNX provider + model auto-selection."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import onnxruntime

log = logging.getLogger("neural_tts_provider_kokoro_onnx.engine")

GPU_PROVIDERS = (
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "CoreMLExecutionProvider",
)
CPU_PROVIDER = "CPUExecutionProvider"


def _data_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base) if base else Path.home() / ".local/share"
    return root / "neural-tts-daemon"


def select_provider() -> tuple[str, bool]:
    """Returns (provider_name, using_gpu)."""
    available = set(onnxruntime.get_available_providers())
    for p in GPU_PROVIDERS:
        if p in available:
            return p, True
    return CPU_PROVIDER, False


def resolve_model_paths(using_gpu: bool) -> tuple[Path, Path]:
    data = _data_dir() / "models"
    model_override = os.environ.get("TTS_KOKORO_MODEL_PATH")
    voices_override = os.environ.get("TTS_KOKORO_VOICES_PATH")

    if model_override:
        model_path = Path(model_override)
    elif using_gpu:
        model_path = data / "kokoro-v1.0.fp16-gpu.onnx"
    else:
        model_path = data / "kokoro-v1.0.int8.onnx"
    voices_path = Path(voices_override) if voices_override else data / "voices-v1.0.bin"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Kokoro model not found at {model_path}; "
            f"run `mise run download-models kokoro-onnx` to fetch it"
        )
    if not voices_path.exists():
        raise FileNotFoundError(
            f"Kokoro voices file not found at {voices_path}; "
            f"run `mise run download-models kokoro-onnx` to fetch it"
        )
    return model_path, voices_path


def build_kokoro():
    """Construct the Kokoro instance with auto-selected provider + model."""
    from kokoro_onnx import Kokoro  # type: ignore[import-not-found]

    provider, using_gpu = select_provider()
    model_path, voices_path = resolve_model_paths(using_gpu)
    log.info(
        "provider=%s model=%s voices=%s using_gpu=%s",
        provider,
        model_path,
        voices_path,
        using_gpu,
    )
    # kokoro-onnx >=0.4 accepts `providers=`; fall back if older.
    try:
        return Kokoro(str(model_path), str(voices_path), providers=[provider])
    except TypeError:
        return Kokoro(str(model_path), str(voices_path))
