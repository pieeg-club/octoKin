"""Wire protocol for the PiEEG Octopus-16 wireless dry-electrode sensor.

Hardware spec (2 × TI ADS131M08 + XIAO ESP32-S3 over BLE 5):
  - Service UUID:        4fafc201-1fb5-459e-8fcc-c5c9c331914b
  - Data characteristic: beb5483e-36e1-4688-b7f5-ea07361b26a8 (notify)
  - Cmd characteristic:  c0de0001-36e1-4688-b7f5-ea07361b26a8 (write + notify)
  - Frame format (51 B): 0xA0 | counter (1B) | 16 × int24_be (48B) | 0xC0
  - Scale:               1.2 V ref, gain 4, 24-bit signed -> ~0.03576279 µV/count
  - Sample rate:         250.0 SPS (8.192 MHz clock) or ~244.14 SPS (8.000 MHz clock)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
DATA_CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
COMMAND_CHAR_UUID = "c0de0001-36e1-4688-b7f5-ea07361b26a8"

KNOWN_BLE_NAME_PREFIXES: tuple[str, ...] = ("Octopus-16", "Octopus_16", "bioron_16")

PACKET_SIZE = 51
HEADER_BYTE = 0xA0
FOOTER_BYTE = 0xC0
NUM_CHANNELS = 16
NOMINAL_SPS = 250.0

VREF_VOLTS = 1.2
ADC_GAIN = 4.0
INT24_MAX_POSITIVE = 0x7FFFFF  # 8,388,607
INT24_MIN_NEGATIVE = -0x800000  # -8,388,608
SCALE_UV_PER_COUNT = (VREF_VOLTS / ADC_GAIN / INT24_MAX_POSITIVE) * 1_000_000.0
RAIL_THRESHOLD_UV = INT24_MAX_POSITIVE * SCALE_UV_PER_COUNT * 0.95  # ~285,000 µV


@dataclass(frozen=True, slots=True)
class OctopusSample:
    """One 16-channel Octopus-16 sample decoded into microvolts."""

    timestamp: float
    counter: int
    channels_uv: tuple[float, ...]

    def to_packet(self) -> bytes:
        """Encode sample back into a canonical 51-byte Octopus-16 frame."""
        return encode_packet(self.channels_uv, counter=self.counter)


def decode_int24_be(b0: int, b1: int, b2: int) -> int:
    """Decode big-endian signed 24-bit two's complement integer."""
    raw = (b0 << 16) | (b1 << 8) | b2
    return raw - 0x1000000 if (raw & 0x800000) else raw


def encode_int24_be(value: int) -> bytes:
    """Clamp and encode signed integer into 3-byte big-endian two's complement."""
    clamped = max(INT24_MIN_NEGATIVE, min(INT24_MAX_POSITIVE, int(value)))
    if clamped < 0:
        clamped += 0x1000000
    return bytes(((clamped >> 16) & 0xFF, (clamped >> 8) & 0xFF, clamped & 0xFF))


def decode_packet(
    data: bytes | bytearray | memoryview,
    timestamp: float | None = None,
) -> OctopusSample | None:
    """Parse a 51-byte BLE notification into an OctopusSample, or None if malformed."""
    if len(data) != PACKET_SIZE or data[0] != HEADER_BYTE or data[50] != FOOTER_BYTE:
        return None

    ts = time.time() if timestamp is None else float(timestamp)
    counter = int(data[1])
    uv = []
    for ch in range(NUM_CHANNELS):
        offset = 2 + ch * 3
        counts = decode_int24_be(data[offset], data[offset + 1], data[offset + 2])
        uv.append(counts * SCALE_UV_PER_COUNT)

    return OctopusSample(timestamp=ts, counter=counter, channels_uv=tuple(uv))


def encode_packet(channels_uv: list[float] | tuple[float, ...], counter: int = 0) -> bytes:
    """Pack 16 microvolt channel values into a 51-byte Octopus-16 wire packet."""
    if len(channels_uv) != NUM_CHANNELS:
        raise ValueError(f"Expected {NUM_CHANNELS} channels, got {len(channels_uv)}")

    buf = bytearray(PACKET_SIZE)
    buf[0] = HEADER_BYTE
    buf[1] = counter & 0xFF
    for ch, val_uv in enumerate(channels_uv):
        counts = round(float(val_uv) / SCALE_UV_PER_COUNT)
        offset = 2 + ch * 3
        buf[offset : offset + 3] = encode_int24_be(counts)
    buf[50] = FOOTER_BYTE
    return bytes(buf)


class PacketTracker:
    """Tracks BLE rolling 8-bit sample counter (0-255) and estimates real sample rate.

    Octopus-16 boards fitted with an 8.000 MHz oscillator run at ~244.14 SPS
    while 8.192 MHz boards run at 250.0 SPS; tracking arrival rate prevents drift
    when synchronizing with robot control loops.
    """

    def __init__(self, window_packets: int = 500, nominal_sps: float = NOMINAL_SPS):
        self._window = max(20, int(window_packets))
        self._nominal_sps = float(nominal_sps)
        self._last_counter: int | None = None
        self.received_packets: int = 0
        self.dropped_packets: int = 0
        self._recent_losses: deque[int] = deque(maxlen=self._window)
        self._arrival_times: deque[float] = deque(maxlen=self._window)

    def reset(self) -> None:
        self._last_counter = None
        self.received_packets = 0
        self.dropped_packets = 0
        self._recent_losses.clear()
        self._arrival_times.clear()

    def update(self, counter: int, timestamp: float) -> int:
        """Record a packet arrival and return newly detected dropped packet count."""
        counter &= 0xFF
        dropped_now = 0
        if self._last_counter is not None:
            expected = (self._last_counter + 1) & 0xFF
            if counter != expected:
                dropped_now = (counter - self._last_counter - 1) & 0xFF
                self.dropped_packets += dropped_now

        self._last_counter = counter
        self.received_packets += 1
        self._recent_losses.append(dropped_now)
        self._arrival_times.append(timestamp)
        return dropped_now

    @property
    def packet_loss_rate(self) -> float:
        """Rolling packet loss ratio in [0.0, 1.0]."""
        if not self._recent_losses:
            return 0.0
        lost = sum(self._recent_losses)
        total = len(self._recent_losses) + lost
        return float(lost / total) if total > 0 else 0.0

    @property
    def estimated_sps(self) -> float:
        """Estimated effective sample rate (SPS), falling back to nominal during warm-up."""
        if len(self._arrival_times) < 50:
            return self._nominal_sps
        dt = self._arrival_times[-1] - self._arrival_times[0]
        if dt <= 1e-6:
            return self._nominal_sps
        samples_elapsed = (len(self._arrival_times) - 1) + sum(list(self._recent_losses)[1:])
        sps = samples_elapsed / dt
        if 150.0 <= sps <= 400.0:
            return float(sps)
        return self._nominal_sps
