"""MOSS-TTS-Nano ONNX runtime setup.

Imports `onnx_tts_runtime.OnnxTtsRuntime` from the upstream repo vendored by
scripts/install_provider.py at the pin in scripts/_moss_tts_nano_pin.py. Adds
the vendor dir to sys.path before importing (upstream uses setuptools with
top-level py-modules — without the path insert the import will fail even
inside the provider venv).

Reuses kokoro's GPU/CPU EP auto-select pattern.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("neural_tts_provider_moss_tts_nano.engine")

# providers/moss-tts-nano/vendor/MOSS-TTS-Nano/ — three parents up from this
# file: .../src/neural_tts_provider_moss_tts_nano/engine.py → src → project_dir.
_VENDOR_DIR = Path(__file__).resolve().parents[2] / "vendor" / "MOSS-TTS-Nano"

MODELS_SUBDIR = "moss-tts-nano"  # under ~/.local/share/neural-tts-daemon/models/

# onnxruntime provider preference, mirrors kokoro-onnx/engine.py.
_GPU_PROVIDERS = (
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "CoreMLExecutionProvider",
)


def _data_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base) if base else Path.home() / ".local/share"
    return root / "neural-tts-daemon"


def resolve_model_dir() -> Path:
    """Returns the dir containing the snapshot_download'd ONNX repos.

    download_models.py lays them out as:
        <models>/moss-tts-nano/
            tts/        ← OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX contents
            tokenizer/  ← OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX contents

    Upstream's `ensure_browser_onnx_model_dir` accepts a parent dir that
    contains both subdirs; it locates the manifest by walking standard
    relative paths inside `MODEL_MANIFEST_CANDIDATE_RELATIVE_PATHS`. Our
    layout requires aliasing — see _link_upstream_layout below.
    """
    override = os.environ.get("TTS_MOSS_TTS_NANO_MODEL_DIR")
    if override:
        return Path(override)
    return _data_dir() / "models" / MODELS_SUBDIR


def _preload_bundled_cuda_libs() -> int:
    """Force-load CUDA/cuDNN .so files bundled by `nvidia-*-cu12` wheels.

    The wheels drop their .so files under `site-packages/nvidia/<pkg>/lib/`,
    but they don't auto-register with the dynamic linker. ORT's CUDA EP
    `dlopen`s names like `libcublas.so.12` which the runtime then can't find,
    so ORT silently falls back to CPU. Preloading with RTLD_GLOBAL makes the
    symbols resolvable. Best-effort; returns count of libs loaded.
    """
    import ctypes
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


def _select_execution_provider() -> str:
    """Return "cuda" if onnxruntime can use a GPU EP, else "cpu".

    Upstream only accepts "cpu" or "cuda" (see ort_cpu_runtime._normalize_
    execution_provider). ROCm/CoreML providers don't have a documented path
    in the upstream runtime, so we map them to "cpu" with a log note.

    Env-var override: `TTS_MOSS_TTS_NANO_EP=cpu|cuda` forces the choice.
    Useful for benchmarking and for working around a slower-than-CPU GPU
    path on small models (Nano's 100M params can be CPU-faster).
    """
    import onnxruntime as ort

    forced = os.environ.get("TTS_MOSS_TTS_NANO_EP", "").strip().lower()
    if forced in ("cpu", "cuda"):
        log.info("execution provider forced via TTS_MOSS_TTS_NANO_EP=%s", forced)
        return forced
    if forced:
        log.warning("ignoring TTS_MOSS_TTS_NANO_EP=%r (expected 'cpu' or 'cuda')", forced)

    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in available:
        return "cuda"
    for p in _GPU_PROVIDERS:
        if p in available:
            log.info(
                "onnxruntime exposes %s but MOSS-TTS-Nano upstream only routes "
                "to cpu/cuda providers; falling back to CPU",
                p,
            )
            break
    return "cpu"


def _link_upstream_layout(model_root: Path) -> Path:
    """Ensure the model dir matches the layout `OnnxTtsRuntime` expects.

    Upstream walks for the manifest at:
        <root>/browser_poc_manifest.json
        <root>/MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json
        <root>/MOSS-TTS-Nano-ONNX-CPU/browser_poc_manifest.json

    download_models.py puts the assets in <root>/tts/ and <root>/tokenizer/.
    We expose them under the names upstream expects via symlinks (idempotent;
    relinks only if the target is wrong).
    """
    tts_src = model_root / "tts"
    codec_src = model_root / "tokenizer"
    if not tts_src.exists():
        raise FileNotFoundError(
            f"MOSS-TTS-Nano model dir not found at {tts_src}; "
            "run `mise run download-models moss-tts-nano` to fetch it"
        )
    if not codec_src.exists():
        raise FileNotFoundError(
            f"MOSS-Audio-Tokenizer-Nano model dir not found at {codec_src}; "
            "run `mise run download-models moss-tts-nano` to fetch it"
        )

    def _ensure_symlink(link: Path, target: Path) -> None:
        if link.is_symlink():
            if link.resolve() == target.resolve():
                return
            link.unlink()
        elif link.exists():
            # A real dir/file is sitting where we want the symlink. Don't
            # touch it; the user likely placed it there deliberately.
            return
        link.symlink_to(target, target_is_directory=True)

    _ensure_symlink(model_root / "MOSS-TTS-Nano-100M-ONNX", tts_src)
    _ensure_symlink(model_root / "MOSS-Audio-Tokenizer-Nano-ONNX", codec_src)
    return model_root


def _ensure_vendor_on_path() -> None:
    if not _VENDOR_DIR.is_dir():
        raise FileNotFoundError(
            f"MOSS-TTS-Nano upstream not vendored at {_VENDOR_DIR}; "
            "run `mise run install-provider moss-tts-nano` to clone it"
        )
    vendor_str = str(_VENDOR_DIR)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)


def _patched_load_reference_audio(runtime: Any) -> Any:
    """Replace OnnxTtsRuntime._load_reference_audio with a soundfile-based loader.

    Upstream's loader uses `torchaudio.load()`, which in torchaudio>=2.11 routes
    through `torchcodec` (a separate, native-lib-heavy dep we'd otherwise have
    to pull). We use soundfile to read the wav and keep torchaudio only for
    the resample (which doesn't need torchcodec).

    Returns the runtime so this can be chained.
    """
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio.functional as F

    def _load(self: Any, reference_audio_path: Any) -> np.ndarray:
        path = str(Path(reference_audio_path).expanduser().resolve())
        # soundfile returns (samples,) for mono or (samples, channels) for multi.
        data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        # → (channels, samples), matching torchaudio.load's convention.
        waveform = torch.from_numpy(data.T).contiguous().to(torch.float32)

        target_sample_rate = int(self.codec_meta["codec_config"]["sample_rate"])
        target_channels = int(self.codec_meta["codec_config"]["channels"])
        if sample_rate != target_sample_rate:
            waveform = F.resample(waveform, sample_rate, target_sample_rate)
        current_channels = int(waveform.shape[0])
        if current_channels == target_channels:
            pass
        elif current_channels == 1 and target_channels > 1:
            waveform = waveform.repeat(target_channels, 1)
        elif current_channels > 1 and target_channels == 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        else:
            raise ValueError(
                f"unsupported reference audio channel conversion: "
                f"{current_channels} -> {target_channels}"
            )
        return waveform.unsqueeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

    # Bind as an instance method so `self` is passed in correctly.
    import types
    runtime._load_reference_audio = types.MethodType(_load, runtime)
    return runtime


def build_runtime() -> Any:
    """Construct an `OnnxTtsRuntime` from the vendored upstream module.

    Returns the runtime object. Caller uses `runtime.synthesize_single_chunk`,
    `runtime.encode_reference_audio`, `runtime.split_voice_clone_text`, etc.
    """
    _ensure_vendor_on_path()

    model_root = _link_upstream_layout(resolve_model_dir())
    execution_provider = _select_execution_provider()
    cpu_threads = int(os.environ.get("TTS_MOSS_TTS_NANO_CPU_THREADS") or 4)

    if execution_provider == "cuda":
        _preload_bundled_cuda_libs()

    log.info(
        "loading MOSS-TTS-Nano (ep=%s, model_dir=%s, cpu_threads=%d)",
        execution_provider, model_root, cpu_threads,
    )
    # Import lazily — upstream imports torch/torchaudio at module load.
    from onnx_tts_runtime import OnnxTtsRuntime  # type: ignore[import-not-found]

    # Upstream eagerly mkdir's `output_dir` at __init__ time (default sits next
    # to the vendor source, which is read-only under systemd's user-service
    # protections). We never call the top-level `synthesize()` that writes wav
    # files there, so the dir is just a vestigial requirement — point it at a
    # cache path that's always writable.
    output_dir = Path.home() / ".cache" / "neural-tts-daemon" / "moss-tts-nano"
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = OnnxTtsRuntime(
        model_dir=model_root,
        thread_count=cpu_threads,
        do_sample=True,
        sample_mode="fixed",
        execution_provider=execution_provider,
        output_dir=str(output_dir),
    )
    _patched_load_reference_audio(runtime)
    sample_rate = int(runtime.codec_meta["codec_config"]["sample_rate"])
    channels = int(runtime.codec_meta["codec_config"]["channels"])
    log.info("runtime ready: sample_rate=%d, channels=%d", sample_rate, channels)
    return runtime
