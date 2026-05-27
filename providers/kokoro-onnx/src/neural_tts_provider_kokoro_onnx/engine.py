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
    """Returns (provider_name, using_gpu).

    Env-var override: `TTS_KOKORO_EP=cpu|cuda|rocm|coreml` forces the choice.
    Useful for benchmarking and for forcing CPU on a GPU-equipped host.
    """
    forced = os.environ.get("TTS_KOKORO_EP", "").strip().lower()
    _aliases = {
        "cpu": (CPU_PROVIDER, False),
        "cuda": ("CUDAExecutionProvider", True),
        "rocm": ("ROCMExecutionProvider", True),
        "coreml": ("CoreMLExecutionProvider", True),
    }
    if forced in _aliases:
        log.info("execution provider forced via TTS_KOKORO_EP=%s", forced)
        return _aliases[forced]
    if forced:
        log.warning("ignoring TTS_KOKORO_EP=%r (expected cpu/cuda/rocm/coreml)", forced)

    available = set(onnxruntime.get_available_providers())
    for p in GPU_PROVIDERS:
        if p in available:
            return p, True
    return CPU_PROVIDER, False


def resolve_model_paths(using_gpu: bool) -> tuple[Path, Path]:
    """Pick the kokoro model file.

    FP16 is preferred on both CPU and GPU when present: it's the upstream
    "fp16-gpu" file, but on modern CPUs (AVX-512 BF16/FP16) it benchmarks
    ~4-5x faster than the INT8 model, which on CPU triggers a lot of
    dequant/requant overhead. Falls back to INT8 if FP16 wasn't downloaded.
    """
    data = _data_dir() / "models"
    model_override = os.environ.get("TTS_KOKORO_MODEL_PATH")
    voices_override = os.environ.get("TTS_KOKORO_VOICES_PATH")

    fp16_path = data / "kokoro-v1.0.fp16-gpu.onnx"
    int8_path = data / "kokoro-v1.0.int8.onnx"

    if model_override:
        model_path = Path(model_override)
    elif fp16_path.exists():
        model_path = fp16_path
    else:
        model_path = int8_path
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


def _preload_bundled_cuda_libs() -> int:
    """Force-load CUDA/cuDNN .so files bundled by `nvidia-*-cu12` wheels.

    The wheels drop their .so files under `site-packages/nvidia/<pkg>/lib/`,
    but they don't auto-register with the dynamic linker. ORT's CUDA EP
    `dlopen`s names like `libcublas.so.12` which the runtime then can't find,
    so ORT silently falls back to CPU. Preloading with RTLD_GLOBAL makes the
    symbols resolvable.

    Returns the number of libs successfully preloaded. Best-effort: silently
    skips static archives and ABI-variant files.
    """
    import ctypes
    import sys
    from pathlib import Path

    nvidia_root = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia"
    if not nvidia_root.is_dir():
        return 0
    count = 0
    for so in sorted(nvidia_root.glob("*/lib/*.so*")):
        try:
            ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
            count += 1
        except OSError:
            pass
    if count:
        log.info("preloaded %d bundled CUDA/cuDNN libs from %s", count, nvidia_root)
    return count


def build_kokoro():
    """Construct the Kokoro instance with auto-selected provider + model.

    kokoro-onnx 0.5.x has two issues we route around here:
      1. Its GPU auto-detect is broken: `importlib.util.find_spec("onnxruntime-gpu")`
         checks an invalid import name (dashes aren't valid in module names)
         and always returns None, so kokoro defaults to ["CPUExecutionProvider"]
         even when onnxruntime-gpu IS installed.
      2. It dropped the `providers=` constructor kwarg.

    Kokoro honors `ONNX_PROVIDER` as an env-var escape hatch (it overrides the
    above default). We set it transiently around the constructor.

    After construction we read `instance.sess.get_providers()` to detect the
    *other* silent-failure mode: ORT itself falling back to CPU when the GPU
    EP fails to load CUDA libs at runtime.
    """
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

    if using_gpu:
        _preload_bundled_cuda_libs()

    prev_env = os.environ.get("ONNX_PROVIDER")
    os.environ["ONNX_PROVIDER"] = provider
    try:
        instance = Kokoro(str(model_path), str(voices_path))
    finally:
        if prev_env is None:
            os.environ.pop("ONNX_PROVIDER", None)
        else:
            os.environ["ONNX_PROVIDER"] = prev_env

    if using_gpu:
        _warn_if_gpu_silently_fell_back(instance, provider, model_path)
    return instance


def _warn_if_gpu_silently_fell_back(instance, requested_provider: str, model_path: Path) -> None:
    """Loudly warn if ORT silently fell back to CPU at session creation.

    Symptom: requested CUDAExecutionProvider, ORT fails to load libcublas/libcudart,
    silently uses CPU. The model "works" but inference is on CPU. Detection: pull
    `instance.sess` (Kokoro 0.5.x) and compare requested vs. actual providers.
    """
    import onnxruntime as ort

    sess = getattr(instance, "sess", None)
    if not isinstance(sess, ort.InferenceSession):
        log.warning(
            "selected GPU provider %s; could not introspect kokoro's session "
            "(attribute layout changed?) — cannot verify GPU is active",
            requested_provider,
        )
        return
    actual = sess.get_providers()
    if requested_provider not in actual:
        log.error(
            "\n%s\n"
            "!! GPU PROVIDER %s WAS REQUESTED BUT NOT ACTIVE — falling back to CPU !!\n"
            "   Active providers: %s\n"
            "   Most likely cause: ORT couldn't load CUDA/cuDNN shared libs.\n"
            "   Run `mise run install-provider kokoro-onnx --extra gpu` to pull\n"
            "   the required nvidia-*-cu12 wheels, OR install system CUDA 12 +\n"
            "   cuDNN 9. To silence and accept CPU, reinstall without --extra gpu.\n"
            "   Model in use: %s\n"
            "%s",
            "=" * 72, requested_provider, actual, model_path, "=" * 72,
        )
    else:
        log.info("confirmed GPU provider %s is active", requested_provider)
