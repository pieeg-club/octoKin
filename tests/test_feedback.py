"""Unit tests for 16-pin contact quality gating, jaw-clench intervention, and blink triggers."""

from __future__ import annotations

import numpy as np

from octokin.feedback import FeedbackConfig, FeedbackEngine
from octokin.protocol import NUM_CHANNELS, OctopusSample


def test_railed_and_flatline_pins_gate_robot_safely() -> None:
    cfg = FeedbackConfig(min_signal_quality=0.60, safe_fallback_speed_scale=0.0)
    engine = FeedbackEngine(cfg)

    # 10 railed pins + 2 flatline pins + only 4 valid pins -> 25% quality (< 60%)
    last_frame = None
    for i in range(200):
        t = i / 250.0
        clean = 15.0 * np.sin(2.0 * np.pi * 10.0 * t)
        ch = [295_000.0] * 10 + [0.0] * 2 + [clean] * 4
        sample = OctopusSample(timestamp=t, counter=i & 0xFF, channels_uv=tuple(ch))
        last_frame = engine.process(sample)

    assert last_frame is not None
    assert not last_frame.contact_ok
    assert last_frame.signal_quality <= 0.25
    assert last_frame.speed_scale == 0.0
    assert not last_frame.jaw_clench
    assert not last_frame.double_blink
    assert sum(last_frame.good_pin_mask) == 4


def test_jaw_clench_emg_triggers_intervention_and_releases() -> None:
    engine = FeedbackEngine(FeedbackConfig())
    rng = np.random.default_rng(7)

    clench_detected = False
    released_after = False

    for i in range(750):  # 3.0 seconds @ 250 SPS
        t = i / 250.0
        base = 15.0 * np.sin(2.0 * np.pi * 10.0 * t) + rng.normal(0.0, 1.5, NUM_CHANNELS)
        if 1.0 <= t <= 1.6:
            # Voluntary jaw-clench EMG burst (65 Hz)
            base = base + 55.0 * np.sin(2.0 * np.pi * 65.0 * t)
        frame = engine.process(
            OctopusSample(timestamp=t, counter=i & 0xFF, channels_uv=tuple(float(v) for v in base))
        )
        if 1.15 <= t <= 1.55 and frame.jaw_clench:
            clench_detected = True
            assert frame.intervention is True
            assert frame.speed_scale == 0.0
            assert frame.double_blink is False  # clench must never masquerade as blink
        if t >= 2.2 and clench_detected and not frame.jaw_clench:
            released_after = True

    assert clench_detected
    assert released_after


def test_single_sample_glitch_does_not_false_trigger_clench() -> None:
    engine = FeedbackEngine(FeedbackConfig())
    for i in range(300):
        t = i / 250.0
        val = 12.0 * np.sin(2.0 * np.pi * 10.0 * t)
        ch = [val] * NUM_CHANNELS
        if i == 150:
            ch = [180.0] * NUM_CHANNELS  # 1-sample transient spike
        frame = engine.process(OctopusSample(timestamp=t, counter=i & 0xFF, channels_uv=tuple(ch)))
        assert not frame.jaw_clench
        assert not frame.intervention


def test_double_blink_requires_two_peaks_within_window() -> None:
    engine = FeedbackEngine(FeedbackConfig())
    double_blink_times: list[float] = []

    for i in range(1250):  # 5.0 seconds
        t = i / 250.0
        sig = 12.0 * np.sin(2.0 * np.pi * 10.0 * t)
        # Isolated single blink at t=1.0s (must NOT trigger double_blink)
        # Paired double blink at t=3.00s and t=3.32s (MUST trigger once)
        for center in (1.00, 3.00, 3.32):
            d = (t - center) / 0.035
            if abs(d) < 3.5:
                sig += 145.0 * float(np.exp(-0.5 * d * d))

        frame = engine.process(
            OctopusSample(timestamp=t, counter=i & 0xFF, channels_uv=tuple([sig] * NUM_CHANNELS))
        )
        if frame.double_blink:
            double_blink_times.append(t)

    assert len(double_blink_times) == 1
    assert 3.25 <= double_blink_times[0] <= 3.45
