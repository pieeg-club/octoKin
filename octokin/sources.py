"""Signal sources and host-to-container bridge for PiEEG Octopus-16.

Includes:
  - Octopus16BLESource: Direct wireless BLE connection to Octopus-16-XXXX
  - DemoSource: Deterministic 16-channel physiological synthesizer (EEG + clench + blink)
  - UDPBridgeSender / UDPBridgeReceiver: Zero-overhead BLE-to-UDP forwarder so
    Dockerized ROS 2 and NVIDIA Isaac Sim / WSL2 receive live Octopus-16 frames
  - CSVRecorder / CSVReplaySource: Session recording and deterministic replay
"""

from __future__ import annotations

import asyncio
import csv
import socket
import struct
import time
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

from .protocol import (
    DATA_CHAR_UUID,
    KNOWN_BLE_NAME_PREFIXES,
    NOMINAL_SPS,
    NUM_CHANNELS,
    PACKET_SIZE,
    SERVICE_UUID,
    OctopusSample,
    decode_packet,
)

BRIDGE_DEFAULT_HOST = "127.0.0.1"
BRIDGE_DEFAULT_PORT = 16160
BRIDGE_HEADER_MAGIC = b"OK16"
BRIDGE_DGRAM_SIZE = 4 + 8 + PACKET_SIZE  # "OK16" + float64 timestamp + 51B packet = 63 bytes


async def scan_octopus_devices(timeout: float = 6.0) -> list[dict[str, object]]:
    """Discover nearby Octopus-16 sensors over Bluetooth LE."""
    try:
        from bleak import BleakScanner
    except ImportError as exc:
        raise RuntimeError(
            "Bluetooth LE support requires 'bleak'. Install with: pip install 'octokin[ble]'"
        ) from exc

    discovered = await BleakScanner.discover(timeout=timeout, return_adv=True)
    matches: list[dict[str, object]] = []
    target_uuid = SERVICE_UUID.lower()

    for _, (device, adv) in discovered.items():
        name = device.name or adv.local_name or ""
        uuids = [u.lower() for u in (adv.service_uuids or [])]
        is_match = (target_uuid in uuids) or any(
            name.startswith(prefix) for prefix in KNOWN_BLE_NAME_PREFIXES
        )
        if is_match:
            matches.append(
                {
                    "name": name or "Octopus-16",
                    "address": device.address,
                    "rssi": getattr(adv, "rssi", None),
                }
            )
    return matches


class Octopus16BLESource:
    """Async stream of OctopusSample frames directly from an Octopus-16 over BLE."""

    def __init__(
        self,
        address: str | None = None,
        scan_timeout: float = 15.0,
        connect_timeout: float = 15.0,
        reconnect: bool = True,
    ):
        self.address = address
        self.scan_timeout = float(scan_timeout)
        self.connect_timeout = float(connect_timeout)
        self.reconnect = bool(reconnect)
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    async def _find_device(self):
        from bleak import BleakScanner

        if self.address:
            dev = await BleakScanner.find_device_by_address(
                self.address, timeout=self.scan_timeout
            )
            if dev is not None:
                return dev

        target_uuid = SERVICE_UUID.lower()

        def _matches(device, adv) -> bool:
            if self.address and device.address.lower() == self.address.lower():
                return True
            uuids = [u.lower() for u in (adv.service_uuids or [])]
            if target_uuid in uuids:
                return True
            name = device.name or getattr(adv, "local_name", "") or ""
            return any(name.startswith(prefix) for prefix in KNOWN_BLE_NAME_PREFIXES)

        return await BleakScanner.find_device_by_filter(_matches, timeout=self.scan_timeout)

    async def __aiter__(self) -> AsyncIterator[OctopusSample]:
        try:
            from bleak import BleakClient
        except ImportError as exc:
            raise RuntimeError(
                "Bluetooth LE support requires 'bleak'. Install with: pip install 'octokin[ble]'"
            ) from exc

        self._stop = False
        while not self._stop:
            device = await self._find_device()
            if device is None:
                if not self.reconnect:
                    raise TimeoutError(
                        "No Octopus-16 sensor found over BLE. Confirm power bank is connected "
                        "(slow LED blink) and the board is unplugged from USB."
                    )
                await asyncio.sleep(1.0)
                continue

            queue: asyncio.Queue[OctopusSample] = asyncio.Queue(maxsize=1024)
            loop = asyncio.get_running_loop()

            def _on_notify(_char, payload: bytearray) -> None:
                sample = decode_packet(payload, timestamp=time.time())
                if sample is not None:
                    loop.call_soon_threadsafe(
                        lambda s=sample: queue.put_nowait(s) if not queue.full() else None
                    )

            try:
                async with BleakClient(device, timeout=self.connect_timeout) as client:
                    await client.start_notify(DATA_CHAR_UUID, _on_notify)
                    while not self._stop and client.is_connected:
                        try:
                            sample = await asyncio.wait_for(queue.get(), timeout=1.0)
                            yield sample
                        except asyncio.TimeoutError:
                            continue
            except Exception:
                if not self.reconnect or self._stop:
                    raise
                await asyncio.sleep(1.0)


class DemoSource:
    """Synthetic 16-channel Octopus-16 generator for hardware-free ROS 2, Isaac, and CI runs.

    Cycles through realistic physiological phases every 12 seconds by default:
      - 0..4 s:   Calm alpha state (10 Hz dominant -> smooth high robot speed_scale)
      - 4..7 s:   Focused beta state (20 Hz dominant -> alert steady control)
      - 7..8.2 s: Voluntary jaw clench (65 Hz EMG burst -> instant robot halt/intervention)
      - 9.2..9.6s: Voluntary double blink (EOG peaks -> mode/waypoint trigger)
    """

    def __init__(
        self,
        sample_rate: float = NOMINAL_SPS,
        max_samples: int | None = None,
        realtime: bool = True,
        scripted_events: bool = True,
        disconnected_pins: tuple[int, ...] = (),
        seed: int = 42,
    ):
        self.sample_rate = float(sample_rate)
        self.max_samples = max_samples
        self.realtime = bool(realtime)
        self.scripted_events = bool(scripted_events)
        self.disconnected_pins = set(disconnected_pins)
        self._rng = np.random.default_rng(seed)
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def generate_sample(self, index: int, t0: float = 0.0) -> OctopusSample:
        """Deterministically synthesize sample `index`."""
        dt = 1.0 / self.sample_rate
        t = index * dt
        phase = t % 12.0

        # Spatial phase spread across the 16-pin hexagonal cluster
        ch_idx = np.arange(NUM_CHANNELS, dtype=np.float64)
        spatial = 0.85 + 0.15 * np.cos(ch_idx * 0.4)

        # Base physiological rhythms (µV)
        theta = 6.0 * np.sin(2.0 * np.pi * 6.0 * t + ch_idx * 0.05)
        alpha_amp = 22.0 if phase < 4.0 else 7.0
        beta_amp = 6.0 if phase < 4.0 else 16.0
        alpha = alpha_amp * np.sin(2.0 * np.pi * 10.2 * t + ch_idx * 0.03)
        beta = beta_amp * np.sin(2.0 * np.pi * 20.5 * t + ch_idx * 0.08)
        noise = self._rng.normal(0.0, 2.2, size=NUM_CHANNELS)

        signal_uv = spatial * (theta + alpha + beta) + noise

        if self.scripted_events:
            # 1) Jaw-clench EMG burst between 7.0 s and 8.0 s
            if 7.0 <= phase <= 8.0:
                emg = 55.0 * np.sin(2.0 * np.pi * 64.0 * t + ch_idx * 0.3)
                emg += self._rng.normal(0.0, 18.0, size=NUM_CHANNELS)
                signal_uv = signal_uv + emg

            # 2) Double blink centered at 9.20 s and 9.50 s (Gaussian EOG deflections)
            for blink_center in (9.20, 9.50):
                dist = (phase - blink_center) / 0.035
                if abs(dist) < 3.5:
                    signal_uv = signal_uv + 145.0 * np.exp(-0.5 * dist * dist)

        for bad_ch in self.disconnected_pins:
            if 0 <= bad_ch < NUM_CHANNELS:
                signal_uv[bad_ch] = 295_000.0  # Railed floating pin

        return OctopusSample(
            timestamp=t0 + t,
            counter=index & 0xFF,
            channels_uv=tuple(float(v) for v in signal_uv),
        )

    async def __aiter__(self) -> AsyncIterator[OctopusSample]:
        self._stop = False
        t0 = time.time()
        dt = 1.0 / self.sample_rate
        idx = 0
        while not self._stop and (self.max_samples is None or idx < self.max_samples):
            sample = self.generate_sample(idx, t0=t0)
            yield sample
            idx += 1
            if self.realtime:
                target = t0 + idx * dt
                delay = target - time.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    await asyncio.sleep(0)


def pack_bridge_datagram(sample: OctopusSample) -> bytes:
    """Pack timestamp + 51-byte Octopus-16 packet into a 63-byte UDP datagram."""
    return BRIDGE_HEADER_MAGIC + struct.pack(">d", sample.timestamp) + sample.to_packet()


def unpack_bridge_datagram(datagram: bytes) -> OctopusSample | None:
    """Unpack either a 63-byte framed bridge datagram or a raw 51-byte Octopus-16 packet."""
    if len(datagram) == BRIDGE_DGRAM_SIZE and datagram[:4] == BRIDGE_HEADER_MAGIC:
        (ts,) = struct.unpack(">d", datagram[4:12])
        return decode_packet(datagram[12:], timestamp=ts)
    if len(datagram) == PACKET_SIZE:
        return decode_packet(datagram)
    return None


class UDPBridgeSender:
    """Forwards OctopusSample frames over UDP (e.g. Windows/macOS host -> Docker/WSL2/Isaac)."""

    def __init__(self, host: str = BRIDGE_DEFAULT_HOST, port: int = BRIDGE_DEFAULT_PORT):
        self.host = host
        self.port = int(port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, sample: OctopusSample) -> None:
        self._sock.sendto(pack_bridge_datagram(sample), (self.host, self.port))

    def close(self) -> None:
        self._sock.close()

    def __enter__(self) -> UDPBridgeSender:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class _UDPQueueProtocol(asyncio.DatagramProtocol):
    def __init__(self, queue: asyncio.Queue[OctopusSample]):
        self._queue = queue

    def datagram_received(self, data: bytes, _addr) -> None:
        sample = unpack_bridge_datagram(data)
        if sample is not None and not self._queue.full():
            self._queue.put_nowait(sample)


class UDPBridgeReceiver:
    """Async SignalSource receiving Octopus-16 frames forwarded by `octokin bridge`."""

    def __init__(
        self,
        bind_host: str = "0.0.0.0",
        port: int = BRIDGE_DEFAULT_PORT,
        max_samples: int | None = None,
    ):
        self.bind_host = bind_host
        self.port = int(port)
        self.max_samples = max_samples
        self._stop = False
        self.bound_port: int = self.port

    def stop(self) -> None:
        self._stop = True

    async def __aiter__(self) -> AsyncIterator[OctopusSample]:
        self._stop = False
        queue: asyncio.Queue[OctopusSample] = asyncio.Queue(maxsize=2048)
        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(
            lambda: _UDPQueueProtocol(queue),
            local_addr=(self.bind_host, self.port),
        )
        sock = transport.get_extra_info("socket")
        if sock is not None:
            self.bound_port = sock.getsockname()[1]

        count = 0
        try:
            while not self._stop and (self.max_samples is None or count < self.max_samples):
                try:
                    sample = await asyncio.wait_for(queue.get(), timeout=0.25)
                    count += 1
                    yield sample
                except asyncio.TimeoutError:
                    continue
        finally:
            transport.close()


class CSVRecorder:
    """Streams OctopusSample frames to a clean, human-readable CSV file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        headers = ["timestamp", "counter"] + [f"ch{i + 1}_uv" for i in range(NUM_CHANNELS)]
        self._writer.writerow(headers)

    def write(self, sample: OctopusSample) -> None:
        row = [f"{sample.timestamp:.6f}", sample.counter] + [
            f"{v:.4f}" for v in sample.channels_uv
        ]
        self._writer.writerow(row)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> CSVRecorder:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class CSVReplaySource:
    """Replays a recorded Octopus-16 CSV file as an async SignalSource."""

    def __init__(self, path: str | Path, realtime: bool = False):
        self.path = Path(path)
        self.realtime = bool(realtime)

    async def __aiter__(self) -> AsyncIterator[OctopusSample]:
        prev_ts: float | None = None
        with self.path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = float(row["timestamp"])
                counter = int(row["counter"])
                channels = tuple(float(row[f"ch{i + 1}_uv"]) for i in range(NUM_CHANNELS))
                if self.realtime and prev_ts is not None:
                    delay = max(0.0, min(0.5, ts - prev_ts))
                    if delay > 0:
                        await asyncio.sleep(delay)
                prev_ts = ts
                yield OctopusSample(timestamp=ts, counter=counter, channels_uv=channels)
