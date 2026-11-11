"""Human physiological feedback engine for robotics and reinforcement learning.

Converts raw 16-channel Octopus-16 samples into robot-safe control primitives:
  - Pin contact quality gate (detects unseated dry pogo pins, rail saturation, flatlines)
  - Fast discrete triggers:
      * Voluntary jaw clench (35–100 Hz EMG burst) -> robot safety pause / intervention
      * Voluntary double blink (1–10 Hz EOG peaks) -> hands-free mode / waypoint trigger
  - Smooth continuous state:
      * Alpha relaxation ratio & Beta/(Alpha+Theta) engagement index
      * Bounded, contact-gated robot velocity scaling (`speed_scale`)
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

import numpy as np

from .dsp import BANDS, RealtimeDSP
from .protocol import NOMINAL_SPS, NUM_CHANNELS, RAIL_THRESHOLD_UV, OctopusSample, PacketTracker


@dataclass(slots=True)
class FeedbackConfig:
    """Configuration thresholds for contact gating, triggers, and velocity modulation."""

    sample_rate: float = NOMINAL_SPS
    notch_hz: float | None = 50.0

    # Contact quality gating (16 dry pogo pins)
    quality_window_samples: int = 125  # 0.5 s @ 250 SPS
    flatline_std_uv: float = 0.05
    max_clean_eeg_std_uv: float = 220.0
    min_signal_quality: float = 0.60
    safe_fallback_speed_scale: float = 0.0

    # Jaw-clench EMG intervention detector (35–100 Hz RMS)
    emg_window_samples: int = 25  # 100 ms sliding RMS
    clench_rms_on_uv: float = 28.0
    clench_rms_off_uv: float = 16.0
    clench_min_channels: int = 4
    clench_hold_samples: int = 12  # ~48 ms sustained burst required

    # Double-blink EOG detector
    blink_peak_uv: float = 95.0
    blink_max_emg_uv: float = 24.0  # reject jaw clenches masquerading as blinks
    blink_min_channel_fraction: float = 0.25
    blink_refractory_s: float = 0.16
    double_blink_window_s: float = 0.75
    double_blink_cooldown_s: float = 1.0

    # Spectral update & continuous velocity scaling
    spectrum_stride_samples: int = 10  # 25 Hz spectral/state update @ 250 SPS
    min_speed_scale: float = 0.25
    max_speed_scale: float = 1.00
    speed_ema_alpha: float = 0.15

    @property
    def blink_min_Channel_fraction(self) -> float:
        """Backward-compatible alias for blink_min_channel_fraction."""
        return self.blink_min_channel_fraction


@dataclass(frozen=True, slots=True)
class HumanFeedbackFrame:
    """Robot-ready physiological feedback snapshot emitted by FeedbackEngine."""

    timestamp: float
    sample_index: int
    estimated_sps: float
    packet_loss_rate: float

    # Contact quality
    signal_quality: float
    contact_ok: bool
    good_pin_mask: tuple[bool, ...]

    # Discrete physiological triggers
    intervention: bool  # True during jaw clench OR unsafe signal dropout if configured
    jaw_clench: bool
    double_blink: bool
    emg_rms_uv: float

    # Continuous physiological state & robot velocity modulation
    alpha_ratio: float
    engagement_index: float
    speed_scale: float
    band_powers: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))


class FeedbackEngine:
    """Stateful processor turning OctopusSample streams into HumanFeedbackFrames."""

    def __init__(self, config: FeedbackConfig | None = None):
        self.config = config or FeedbackConfig()
        self.dsp = RealtimeDSP(
            num_channels=NUM_CHANNELS,
            fs=self.config.sample_rate,
            notch_hz=self.config.notch_hz,
        )
        self.tracker = PacketTracker(nominal_sps=self.config.sample_rate)

        self._sample_index = 0
        self._q_cap = max(1, int(self.config.quality_window_samples))
        self._emg_cap = max(1, int(self.config.emg_window_samples))
        self._raw_ring = np.zeros((self._q_cap, NUM_CHANNELS), dtype=np.float64)
        self._eeg_ring = np.zeros((self._q_cap, NUM_CHANNELS), dtype=np.float64)
        self._emg_sq_ring = np.zeros((self._emg_cap, NUM_CHANNELS), dtype=np.float64)
        self._q_len = 0
        self._q_pos = 0
        self._emg_len = 0
        self._emg_pos = 0

        self._clench_active = False
        self._clench_streak = 0

        self._last_single_blink_ts: float = -100.0
        self._pending_blink_ts: float | None = None
        self._last_double_blink_ts: float = -100.0

        self._band_powers: dict[str, float] = {b: 0.0 for b in BANDS}
        self._alpha_ratio: float = 0.33
        self._engagement_index: float = 0.50
        self._smoothed_speed: float = 0.65

    def reset(self) -> None:
        self.dsp.reset()
        self.tracker.reset()
        self._sample_index = 0
        self._raw_ring.fill(0.0)
        self._eeg_ring.fill(0.0)
        self._emg_sq_ring.fill(0.0)
        self._q_len = 0
        self._q_pos = 0
        self._emg_len = 0
        self._emg_pos = 0
        self._clench_active = False
        self._clench_streak = 0
        self._last_single_blink_ts = -100.0
        self._pending_blink_ts = None
        self._last_double_blink_ts = -100.0
        self._band_powers = {b: 0.0 for b in BANDS}
        self._alpha_ratio = 0.33
        self._engagement_index = 0.50
        self._smoothed_speed = 0.65

    def calibrate(self, samples: Iterable[OctopusSample]) -> dict[str, float]:
        """Calibrate EMG/EOG trigger thresholds from a resting baseline sample stream."""
        emg_medians: list[float] = []
        eeg_p95s: list[float] = []
        for s in samples:
            frame = self.process(s)
            if frame.contact_ok and not frame.jaw_clench:
                emg_medians.append(frame.emg_rms_uv)
                good_mask = np.asarray(frame.good_pin_mask, dtype=bool)
                if np.any(good_mask) and self._q_len > 0:
                    latest_idx = (self._q_pos - 1) % self._q_cap
                    eeg_abs = np.abs(self._eeg_ring[latest_idx, good_mask])
                    eeg_p95s.append(float(np.percentile(eeg_abs, 95)))

        if emg_medians:
            baseline_emg = float(np.median(emg_medians))
            self.config.clench_rms_on_uv = round(max(18.0, baseline_emg * 4.5), 2)
            self.config.clench_rms_off_uv = round(
                max(10.0, self.config.clench_rms_on_uv * 0.57), 2
            )
            self.config.blink_max_emg_uv = round(
                max(15.0, self.config.clench_rms_on_uv * 0.85), 2
            )
        if eeg_p95s:
            baseline_eeg_peak = float(np.median(eeg_p95s))
            self.config.blink_peak_uv = round(max(65.0, baseline_eeg_peak * 4.0), 2)

        return {
            "clench_rms_on_uv": self.config.clench_rms_on_uv,
            "clench_rms_off_uv": self.config.clench_rms_off_uv,
            "blink_max_emg_uv": self.config.blink_max_emg_uv,
            "blink_peak_uv": self.config.blink_peak_uv,
        }

    def _evaluate_contact(self) -> tuple[np.ndarray, float]:
        """Return (good_pin_bool_array, overall_signal_quality_0_to_1)."""
        raw_arr = self._raw_ring[: self._q_len]
        eeg_arr = self._eeg_ring[: self._q_len]

        not_railed = np.max(np.abs(raw_arr), axis=0) < RAIL_THRESHOLD_UV
        if self._q_len < 16:
            good_pins = not_railed
        else:
            eeg_std = np.std(eeg_arr, axis=0)
            not_flatline = eeg_std >= self.config.flatline_std_uv
            # Raise noise ceiling during active jaw clench so clenches don't trip bad-pin gate
            ceiling = (
                self.config.max_clean_eeg_std_uv * 2.5
                if self._clench_active
                else self.config.max_clean_eeg_std_uv
            )
            not_noisy = eeg_std <= ceiling
            good_pins = not_railed & not_flatline & not_noisy

        pin_score = float(np.mean(good_pins))
        loss_penalty = min(1.0, self.tracker.packet_loss_rate * 2.0)
        quality = max(0.0, min(1.0, pin_score * (1.0 - loss_penalty)))
        return good_pins, quality

    def _detect_jaw_clench(self, good_pins: np.ndarray) -> tuple[bool, float]:
        """Detect sustained high-frequency EMG burst across seated pins."""
        if self._emg_len == 0 or int(np.sum(good_pins)) == 0:
            self._clench_streak = 0
            self._clench_active = False
            return False, 0.0

        emg_rms_per_ch = np.sqrt(np.mean(self._emg_sq_ring[: self._emg_len], axis=0))
        valid_rms = emg_rms_per_ch[good_pins]
        median_rms = float(np.median(valid_rms))

        if not self._clench_active:
            active_ch = int(np.sum(valid_rms >= self.config.clench_rms_on_uv))
            if active_ch >= min(self.config.clench_min_channels, len(valid_rms)):
                self._clench_streak += 1
                if self._clench_streak >= self.config.clench_hold_samples:
                    self._clench_active = True
            else:
                self._clench_streak = 0
        else:
            active_ch = int(np.sum(valid_rms >= self.config.clench_rms_off_uv))
            if active_ch < max(1, min(self.config.clench_min_channels // 2, len(valid_rms))):
                self._clench_active = False
                self._clench_streak = 0

        return self._clench_active, median_rms

    def _detect_double_blink(
        self,
        eeg_uv: np.ndarray,
        emg_rms_uv: float,
        good_pins: np.ndarray,
        timestamp: float,
    ) -> bool:
        """Detect two distinct EOG blink deflections within double_blink_window_s."""
        num_good = int(np.sum(good_pins))
        if num_good == 0 or emg_rms_uv > self.config.blink_max_emg_uv or self._clench_active:
            return False

        # Expire stale single blink
        if (
            self._pending_blink_ts is not None
            and (timestamp - self._pending_blink_ts) > self.config.double_blink_window_s
        ):
            self._pending_blink_ts = None

        peaking_pins = int(np.sum(np.abs(eeg_uv[good_pins]) >= self.config.blink_peak_uv))
        required_pins = max(2, int(np.ceil(num_good * self.config.blink_min_channel_fraction)))
        if peaking_pins < required_pins:
            return False

        if (timestamp - self._last_single_blink_ts) < self.config.blink_refractory_s:
            return False

        self._last_single_blink_ts = timestamp
        if (timestamp - self._last_double_blink_ts) < self.config.double_blink_cooldown_s:
            return False

        if self._pending_blink_ts is None:
            self._pending_blink_ts = timestamp
            return False

        # Second blink arrived inside valid double-blink interval
        self._pending_blink_ts = None
        self._last_double_blink_ts = timestamp
        return True

    def process(self, sample: OctopusSample) -> HumanFeedbackFrame:
        """Ingest one OctopusSample and return the updated HumanFeedbackFrame."""
        self.tracker.update(sample.counter, sample.timestamp)
        filtered = self.dsp.step(sample.channels_uv)

        self._raw_ring[self._q_pos] = sample.channels_uv
        self._eeg_ring[self._q_pos] = filtered.eeg_uv
        self._q_pos = (self._q_pos + 1) % self._q_cap
        if self._q_len < self._q_cap:
            self._q_len += 1

        self._emg_sq_ring[self._emg_pos] = filtered.emg_uv**2
        self._emg_pos = (self._emg_pos + 1) % self._emg_cap
        if self._emg_len < self._emg_cap:
            self._emg_len += 1

        self._sample_index += 1

        good_pins, signal_quality = self._evaluate_contact()
        contact_ok = signal_quality >= self.config.min_signal_quality

        if contact_ok:
            jaw_clench, emg_rms = self._detect_jaw_clench(good_pins)
            double_blink = self._detect_double_blink(
                filtered.eeg_uv, emg_rms, good_pins, sample.timestamp
            )
        else:
            self._clench_active = False
            self._clench_streak = 0
            jaw_clench = False
            emg_rms = 0.0
            double_blink = False

        # Recompute spectral state periodically once FFT window is primed
        if (
            contact_ok
            and (self._sample_index % self.config.spectrum_stride_samples == 0)
            and self.dsp.spectrum_ready
        ):
            powers = self.dsp.compute_band_powers(active_mask=good_pins)
            if powers is not None:
                self._band_powers = powers
                eps = 1e-6
                theta = powers["theta"]
                alpha = powers["alpha"]
                beta = powers["beta"]
                self._alpha_ratio = float(alpha / (theta + alpha + beta + eps))
                self._engagement_index = float(beta / (alpha + theta + eps))

                # Calm/focused steady control scales smoothly within [min_speed, max_speed]
                norm_drive = float(np.clip(0.35 + 0.65 * (1.0 - self._alpha_ratio), 0.0, 1.0))
                target_speed = self.config.min_speed_scale + norm_drive * (
                    self.config.max_speed_scale - self.config.min_speed_scale
                )
                a = self.config.speed_ema_alpha
                self._smoothed_speed = (1.0 - a) * self._smoothed_speed + a * target_speed

        # Safety arbitration: jaw clench halts immediately; bad contact falls back safely
        if jaw_clench:
            speed_scale = 0.0
            intervention = True
        elif not contact_ok:
            speed_scale = self.config.safe_fallback_speed_scale
            intervention = False
        else:
            speed_scale = float(
                np.clip(
                    self._smoothed_speed,
                    self.config.min_speed_scale,
                    self.config.max_speed_scale,
                )
            )
            intervention = False

        return HumanFeedbackFrame(
            timestamp=sample.timestamp,
            sample_index=self._sample_index,
            estimated_sps=round(self.tracker.estimated_sps, 2),
            packet_loss_rate=round(self.tracker.packet_loss_rate, 4),
            signal_quality=round(signal_quality, 3),
            contact_ok=contact_ok,
            good_pin_mask=tuple(bool(v) for v in good_pins),
            intervention=intervention,
            jaw_clench=jaw_clench,
            double_blink=double_blink,
            emg_rms_uv=round(emg_rms, 2),
            alpha_ratio=round(self._alpha_ratio, 4),
            engagement_index=round(self._engagement_index, 4),
            speed_scale=round(speed_scale, 3),
            band_powers={k: round(v, 4) for k, v in self._band_powers.items()},
        )
