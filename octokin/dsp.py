"""Stateful real-time DSP pipeline for 16-channel Octopus-16 biosignals.

Separates incoming raw microvolt streams into:
  1. Powerline-notched stream (50 or 60 Hz + 2nd harmonic)
  2. Clean EEG/EOG band (1.0–45.0 Hz) for spectral band powers and blink detection
  3. High-frequency EMG band (35.0–100.0 Hz) for fast, drift-immune jaw-clench detection
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .protocol import NOMINAL_SPS, NUM_CHANNELS, RAIL_THRESHOLD_UV

BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


@dataclass(frozen=True, slots=True)
class FilteredStep:
    """Single-sample output from RealtimeDSP across all channels."""

    notched_uv: np.ndarray
    eeg_uv: np.ndarray
    emg_uv: np.ndarray


class RealtimeDSP:
    """Vectorized multichannel IIR filter bank + windowed Welch band-power estimator."""

    def __init__(
        self,
        num_channels: int = NUM_CHANNELS,
        fs: float = NOMINAL_SPS,
        notch_hz: float | None = 50.0,
        notch_q: float = 30.0,
        eeg_band: tuple[float, float] = (1.0, 45.0),
        emg_band: tuple[float, float] = (35.0, 100.0),
        fft_size: int = 512,
    ):
        self.num_channels = int(num_channels)
        self.fs = float(fs)
        self.fft_size = int(fft_size)
        nyquist = 0.5 * self.fs

        # 1. Powerline notch cascade (fundamental + 2nd harmonic if safely below Nyquist)
        notch_sections = []
        if notch_hz is not None and 10.0 < notch_hz < nyquist - 2.0:
            b1, a1 = signal.iirnotch(notch_hz, notch_q, fs=self.fs)
            notch_sections.append(signal.tf2sos(b1, a1))
            harmonic = 2.0 * notch_hz
            if harmonic < nyquist - 3.0:
                b2, a2 = signal.iirnotch(harmonic, notch_q, fs=self.fs)
                notch_sections.append(signal.tf2sos(b2, a2))

        self._notch_sos: np.ndarray | None = (
            np.vstack(notch_sections) if notch_sections else None
        )
        self._eeg_sos = signal.butter(
            4,
            [eeg_band[0], min(eeg_band[1], nyquist - 1.0)],
            btype="band",
            fs=self.fs,
            output="sos",
        )
        self._emg_sos = signal.butter(
            4,
            [emg_band[0], min(emg_band[1], nyquist - 1.0)],
            btype="band",
            fs=self.fs,
            output="sos",
        )

        self._primed = False
        self._prev_railed = np.zeros(self.num_channels, dtype=bool)
        self._notch_zi = (
            np.zeros((self._notch_sos.shape[0], self.num_channels, 2), dtype=np.float64)
            if self._notch_sos is not None
            else None
        )
        self._eeg_zi = np.zeros((self._eeg_sos.shape[0], self.num_channels, 2), dtype=np.float64)
        self._emg_zi = np.zeros((self._emg_sos.shape[0], self.num_channels, 2), dtype=np.float64)

        # Rolling ring buffer for spectral estimation (num_channels, fft_size)
        self._eeg_ring = np.zeros((self.num_channels, self.fft_size), dtype=np.float64)
        self._ring_filled = 0

        self._hanning = np.hanning(self.fft_size)
        self._freqs = np.fft.rfftfreq(self.fft_size, d=1.0 / self.fs)
        self._psd_norm = self.fs * float(np.sum(self._hanning**2))
        self._band_masks = {
            name: (self._freqs >= lo) & (self._freqs < hi) for name, (lo, hi) in BANDS.items()
        }

    def reset(self) -> None:
        self._primed = False
        self._prev_railed.fill(False)
        if self._notch_zi is not None:
            self._notch_zi.fill(0.0)
        self._eeg_zi.fill(0.0)
        self._emg_zi.fill(0.0)
        self._eeg_ring.fill(0.0)
        self._ring_filled = 0

    def _prime_channels(self, sample: np.ndarray, mask: np.ndarray) -> None:
        """Seed IIR filter states on selected channels to prevent rail/DC step ringing."""
        vals = sample[mask]
        if self._notch_sos is not None and self._notch_zi is not None:
            zi_unit = signal.sosfilt_zi(self._notch_sos)
            self._notch_zi[:, mask, :] = zi_unit[:, None, :] * vals[None, :, None]
        eeg_unit = signal.sosfilt_zi(self._eeg_sos)
        self._eeg_zi[:, mask, :] = eeg_unit[:, None, :] * vals[None, :, None]
        emg_unit = signal.sosfilt_zi(self._emg_sos)
        self._emg_zi[:, mask, :] = emg_unit[:, None, :] * vals[None, :, None]

    def step(self, channels_uv: tuple[float, ...] | list[float] | np.ndarray) -> FilteredStep:
        """Process one multi-channel sample in microvolts."""
        x = np.asarray(channels_uv, dtype=np.float64)
        if x.shape != (self.num_channels,):
            raise ValueError(f"Expected ({self.num_channels},) sample, got {x.shape}")

        now_railed = np.abs(x) >= RAIL_THRESHOLD_UV
        if not self._primed:
            self._prime_channels(x, np.ones(self.num_channels, dtype=bool))
            self._primed = True
        else:
            # Re-prime any pin that just transitioned from floating rail saturation onto scalp
            recovered = self._prev_railed & (~now_railed)
            if np.any(recovered):
                self._prime_channels(x, recovered)
        self._prev_railed = now_railed

        col = x[:, None]
        if self._notch_sos is not None and self._notch_zi is not None:
            notched_col, self._notch_zi = signal.sosfilt(
                self._notch_sos, col, axis=-1, zi=self._notch_zi
            )
        else:
            notched_col = col

        eeg_col, self._eeg_zi = signal.sosfilt(
            self._eeg_sos, notched_col, axis=-1, zi=self._eeg_zi
        )
        emg_col, self._emg_zi = signal.sosfilt(
            self._emg_sos, notched_col, axis=-1, zi=self._emg_zi
        )

        notched = notched_col[:, 0]
        eeg = eeg_col[:, 0]
        emg = emg_col[:, 0]

        # Shift ring buffer by 1 column and append latest clean EEG sample
        self._eeg_ring[:, :-1] = self._eeg_ring[:, 1:]
        self._eeg_ring[:, -1] = eeg
        if self._ring_filled < self.fft_size:
            self._ring_filled += 1

        return FilteredStep(notched_uv=notched, eeg_uv=eeg, emg_uv=emg)

    @property
    def spectrum_ready(self) -> bool:
        return self._ring_filled >= self.fft_size

    def compute_band_powers(
        self, active_mask: np.ndarray | None = None
    ) -> dict[str, float] | None:
        """Compute mean single-sided PSD (µV²/Hz) per canonical EEG band across good channels."""
        if not self.spectrum_ready:
            return None

        data = self._eeg_ring
        if active_mask is not None:
            if not np.any(active_mask):
                return None
            data = data[active_mask]

        windowed = data * self._hanning[None, :]
        psd = (np.abs(np.fft.rfft(windowed, axis=-1)) ** 2) / self._psd_norm
        psd[:, 1:-1] *= 2.0  # single-sided spectrum correction

        mean_psd = np.mean(psd, axis=0)
        return {
            band: float(np.mean(mean_psd[mask])) if np.any(mask) else 0.0
            for band, mask in self._band_masks.items()
        }
