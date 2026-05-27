"""Streaming resampler. No-op when input/output rates match."""

from __future__ import annotations

from typing import Callable

import numpy as np


class Resampler:
    """Stateful chunk resampler. `process(chunk)` returns possibly-empty resampled chunk.

    Call `flush()` at end-of-stream to drain the last few samples.
    """

    def __init__(self, in_rate: int, out_rate: int):
        self.in_rate = in_rate
        self.out_rate = out_rate
        self._impl: Callable[[np.ndarray], np.ndarray]
        self._flush: Callable[[], np.ndarray]
        if in_rate == out_rate:
            self._impl = lambda x: x
            self._flush = lambda: np.empty(0, dtype=np.float32)
        else:
            import soxr  # type: ignore[import-not-found]
            self._stream = soxr.ResampleStream(
                in_rate=in_rate,
                out_rate=out_rate,
                num_channels=1,
                dtype="float32",
            )
            self._impl = lambda x: self._stream.resample_chunk(x, last=False)
            self._flush = lambda: self._stream.resample_chunk(
                np.empty(0, dtype=np.float32), last=True
            )

    def process(self, samples: np.ndarray) -> np.ndarray:
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32, copy=False)
        if samples.ndim != 1:
            samples = samples.reshape(-1)
        return self._impl(samples)

    def flush(self) -> np.ndarray:
        return self._flush()


def float32_to_s16le_bytes(samples: np.ndarray) -> bytes:
    """Convert a float32 buffer in [-1, 1] to little-endian s16 bytes."""
    if samples.size == 0:
        return b""
    y = np.multiply(samples, 32767.0, dtype=np.float32)
    np.clip(y, -32768.0, 32767.0, out=y)
    return y.astype("<i2", copy=False).tobytes()
