"""Unit tests for stateful multichannel IIR filtering and spectral estimation."""

from __future__ import annotations

import numpy as np

from octokin.dsp import RealtimeDSP
from octokin.protocol import NUM_CHANNELS


def test_dc_offset_priming_prevents_startup_spike() -> None:
    dsp = RealtimeDSP(num_channels=NUM_CHANNELS, fs=250.0, notch_hz=50.0)
    # Feed constant large DC baseline (+12,000 µV) with tiny 10 Hz ripple
    peaks = []
    for i in range(100):
        t = i / 250.0
        val = 12_000.0 + 5.0 * np.sin(2.0 * np.pi * 10.0 * t)
        step = dsp.step([val] * NUM_CHANNELS)
        peaks.append(float(np.max(np.abs(step.eeg_uv))))

    # Without state priming, +12,000 µV DC step rings at thousands of µV
    assert max(peaks) < 25.0


def test_50hz_notch_attenuates_mains_while_preserving_alpha() -> None:
    dsp = RealtimeDSP(num_channels=NUM_CHANNELS, fs=250.0, notch_hz=50.0, fft_size=512)
    alpha_after = []
    mains_only_dsp = RealtimeDSP(num_channels=NUM_CHANNELS, fs=250.0, notch_hz=50.0)
    mains_after = []

    for i in range(750):
        t = i / 250.0
        sig_mixed = 20.0 * np.sin(2.0 * np.pi * 10.0 * t) + 100.0 * np.sin(2.0 * np.pi * 50.0 * t)
        sig_50 = 100.0 * np.sin(2.0 * np.pi * 50.0 * t)
        out_mix = dsp.step([sig_mixed] * NUM_CHANNELS)
        out_50 = mains_only_dsp.step([sig_50] * NUM_CHANNELS)
        if i >= 250:
            alpha_after.append(out_mix.eeg_uv[0])
            mains_after.append(out_50.notched_uv[0])

    # 100 µV @ 50 Hz attenuated by > 30 dB (< 3 µV RMS)
    assert float(np.std(mains_after)) < 2.0
    # 20 µV peak (~14.1 µV RMS) @ 10 Hz preserved cleanly
    assert 12.0 <= float(np.std(alpha_after)) <= 16.0

    powers = dsp.compute_band_powers()
    assert powers is not None
    assert powers["alpha"] > 15.0 * powers["theta"]
    assert powers["alpha"] > 15.0 * powers["beta"]


def test_eeg_vs_emg_band_separation_and_active_mask() -> None:
    dsp = RealtimeDSP(num_channels=NUM_CHANNELS, fs=250.0, notch_hz=50.0, fft_size=512)
    emg_vals = []
    eeg_vals = []

    for i in range(600):
        t = i / 250.0
        # Ch 0..14: 70 Hz jaw-clench EMG; Ch 15: huge corrupt 20 Hz artifact
        emg_70 = 40.0 * np.sin(2.0 * np.pi * 70.0 * t)
        sample = [emg_70] * 15 + [5000.0 * np.sin(2.0 * np.pi * 20.0 * t)]
        step = dsp.step(sample)
        if i >= 200:
            emg_vals.append(step.emg_uv[0])
            eeg_vals.append(step.eeg_uv[0])

    # 70 Hz passes cleanly through EMG branch (35-100 Hz) and is rejected by EEG (1-45 Hz)
    assert float(np.std(emg_vals)) > 24.0
    assert float(np.std(eeg_vals)) < 2.5

    mask = np.array([True] * 15 + [False])
    masked_powers = dsp.compute_band_powers(active_mask=mask)
    unmasked_powers = dsp.compute_band_powers()
    assert masked_powers is not None and unmasked_powers is not None
    assert masked_powers["beta"] < 1.0
    assert unmasked_powers["beta"] > 500.0


def test_rail_recovery_re_primes_cleanly() -> None:
    dsp = RealtimeDSP(num_channels=NUM_CHANNELS, fs=250.0, notch_hz=50.0)
    # 50 samples railed in air before user places Octopus-16 on scalp
    for _ in range(50):
        dsp.step([295_000.0] * NUM_CHANNELS)

    # Immediately touch scalp (~15 µV alpha): filter must not ring at thousands of µV
    post_seat_peaks = []
    for i in range(50):
        t = i / 250.0
        step = dsp.step([15.0 * np.sin(2.0 * np.pi * 10.0 * t)] * NUM_CHANNELS)
        post_seat_peaks.append(float(np.max(np.abs(step.eeg_uv))))

    assert max(post_seat_peaks) < 35.0
