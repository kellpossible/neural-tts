"""Resampler basic correctness."""

from __future__ import annotations

import numpy as np

from kde_tts_daemon.resample import Resampler, float32_to_s16le_bytes


def test_passthrough_when_rates_match():
    r = Resampler(in_rate=24000, out_rate=24000)
    samples = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    out = r.process(samples)
    tail = r.flush()
    assert np.array_equal(out, samples)
    assert tail.size == 0


def test_downsample_48k_to_24k_changes_length():
    r = Resampler(in_rate=48000, out_rate=24000)
    samples = np.zeros(1000, dtype=np.float32)
    out = r.process(samples)
    tail = r.flush()
    total = out.size + tail.size
    # Output is approximately half the input length, with a small soxr filter
    # delay that may push some samples into the tail.
    assert 400 < total < 600


def test_float32_to_s16le_bytes_clipping():
    samples = np.array([0.0, 0.5, 1.5, -2.0], dtype=np.float32)
    out = float32_to_s16le_bytes(samples)
    # 4 samples × 2 bytes each
    assert len(out) == 8
    decoded = np.frombuffer(out, dtype="<i2")
    assert decoded[0] == 0
    assert decoded[1] == int(0.5 * 32767)
    assert decoded[2] == 32767  # clipped
    assert decoded[3] == -32768  # clipped
