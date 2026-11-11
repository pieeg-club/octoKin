"""Tests for DemoSource, UDP host-to-container bridge, and CSV recording/replay."""

from __future__ import annotations

import asyncio
from pathlib import Path

from octokin.feedback import FeedbackEngine
from octokin.protocol import SCALE_UV_PER_COUNT
from octokin.sources import (
    CSVRecorder,
    CSVReplaySource,
    DemoSource,
    UDPBridgeReceiver,
    UDPBridgeSender,
    unpack_bridge_datagram,
)


def test_demo_source_full_physiological_cycle() -> None:
    demo = DemoSource(realtime=False, max_samples=3000, scripted_events=True)
    engine = FeedbackEngine()

    saw_contact = False
    saw_clench = False
    saw_double_blink = False

    for idx in range(3000):  # 12 seconds @ 250 SPS
        frame = engine.process(demo.generate_sample(idx, t0=100.0))
        if frame.contact_ok:
            saw_contact = True
        if frame.jaw_clench:
            saw_clench = True
        if frame.double_blink:
            saw_double_blink = True

    assert saw_contact
    assert saw_clench
    assert saw_double_blink


async def test_udp_bridge_sender_receiver_roundtrip() -> None:
    demo = DemoSource(realtime=False)
    expected = [demo.generate_sample(i, t0=500.0) for i in range(10)]

    receiver = UDPBridgeReceiver(bind_host="127.0.0.1", port=0, max_samples=len(expected))
    received = []

    async def _consume() -> None:
        async for s in receiver:
            received.append(s)

    consumer_task = asyncio.create_task(_consume())
    # Wait briefly until UDP socket binds and populates `bound_port`
    while receiver.bound_port == 0:
        await asyncio.sleep(0.01)

    with UDPBridgeSender(host="127.0.0.1", port=receiver.bound_port) as sender:
        for s in expected:
            sender.send(s)
            await asyncio.sleep(0.002)

    await asyncio.wait_for(consumer_task, timeout=3.0)
    assert len(received) == len(expected)
    for got, want in zip(received, expected, strict=True):
        assert got.counter == want.counter
        assert abs(got.timestamp - want.timestamp) < 1e-6
        assert abs(got.channels_uv[0] - want.channels_uv[0]) <= SCALE_UV_PER_COUNT


def test_unpack_accepts_raw_51b_octopus_packet() -> None:
    demo = DemoSource(realtime=False)
    sample = demo.generate_sample(7, t0=123.0)
    raw_51 = sample.to_packet()
    unpacked = unpack_bridge_datagram(raw_51)
    assert unpacked is not None
    assert unpacked.counter == 7


async def test_csv_recorder_and_replay_roundtrip(tmp_path: Path) -> None:
    csv_file = tmp_path / "session.csv"
    demo = DemoSource(realtime=False)
    original = [demo.generate_sample(i, t0=1000.0) for i in range(25)]

    with CSVRecorder(csv_file) as rec:
        for s in original:
            rec.write(s)

    replayed = [s async for s in CSVReplaySource(csv_file, realtime=False)]
    assert len(replayed) == len(original)
    for got, want in zip(replayed, original, strict=True):
        assert got.counter == want.counter
        assert abs(got.timestamp - want.timestamp) < 1e-5
        assert abs(got.channels_uv[15] - want.channels_uv[15]) < 1e-3
