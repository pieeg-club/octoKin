"""Unit tests for PiEEG Octopus-16 51-byte wire protocol and packet tracking."""

from __future__ import annotations

import pytest

from octokin.protocol import (
    FOOTER_BYTE,
    HEADER_BYTE,
    INT24_MAX_POSITIVE,
    INT24_MIN_NEGATIVE,
    NUM_CHANNELS,
    PACKET_SIZE,
    SCALE_UV_PER_COUNT,
    PacketTracker,
    decode_int24_be,
    decode_packet,
    encode_int24_be,
    encode_packet,
)


def test_int24_signed_extremes_and_clamping() -> None:
    for val in (0, 1, -1, 123456, -654321, INT24_MAX_POSITIVE, INT24_MIN_NEGATIVE):
        encoded = encode_int24_be(val)
        assert len(encoded) == 3
        assert decode_int24_be(encoded[0], encoded[1], encoded[2]) == val

    # Overflow clamping
    hi = encode_int24_be(INT24_MAX_POSITIVE + 5000)
    lo = encode_int24_be(INT24_MIN_NEGATIVE - 5000)
    assert decode_int24_be(*hi) == INT24_MAX_POSITIVE
    assert decode_int24_be(*lo) == INT24_MIN_NEGATIVE


def test_packet_roundtrip_microvolts() -> None:
    expected_uv = [(-8.0 + i) * 15.25 for i in range(NUM_CHANNELS)]
    pkt = encode_packet(expected_uv, counter=203)
    assert len(pkt) == PACKET_SIZE
    assert pkt[0] == HEADER_BYTE
    assert pkt[1] == 203
    assert pkt[50] == FOOTER_BYTE

    decoded = decode_packet(pkt, timestamp=1000.5)
    assert decoded is not None
    assert decoded.timestamp == 1000.5
    assert decoded.counter == 203
    assert len(decoded.channels_uv) == NUM_CHANNELS
    for got, want in zip(decoded.channels_uv, expected_uv, strict=True):
        assert abs(got - want) <= SCALE_UV_PER_COUNT


def test_decode_rejects_malformed_packets() -> None:
    valid = bytearray(encode_packet([0.0] * NUM_CHANNELS, counter=1))
    assert decode_packet(valid[:50]) is None
    assert decode_packet(valid + b"\x00") is None

    bad_hdr = valid.copy()
    bad_hdr[0] = 0xA1
    assert decode_packet(bad_hdr) is None

    bad_ftr = valid.copy()
    bad_ftr[50] = 0xC1
    assert decode_packet(bad_ftr) is None


def test_encode_validates_channel_count() -> None:
    with pytest.raises(ValueError, match="Expected 16 channels"):
        encode_packet([0.0] * 8)


def test_packet_tracker_wraparound_and_loss() -> None:
    tracker = PacketTracker(window_packets=100, nominal_sps=250.0)

    # Clean wraparound 254 -> 255 -> 0 -> 1
    for idx, cnt in enumerate((254, 255, 0, 1)):
        assert tracker.update(cnt, timestamp=idx * 0.004) == 0
    assert tracker.dropped_packets == 0
    assert tracker.packet_loss_rate == 0.0

    # Jump from 1 -> 5 means counters 2, 3, 4 were lost (3 dropped packets)
    dropped = tracker.update(5, timestamp=8 * 0.004)
    assert dropped == 3
    assert tracker.dropped_packets == 3
    assert tracker.received_packets == 5
    # 3 lost out of (5 received + 3 lost) = 0.375
    assert pytest.approx(tracker.packet_loss_rate, rel=1e-4) == 3.0 / 8.0


def test_packet_tracker_estimates_244_vs_250_sps() -> None:
    # Simulate 8.000 MHz Octopus-16 clock (~244.14 SPS) with occasional packet drops
    true_sps = 244.14
    tracker = PacketTracker(window_packets=200, nominal_sps=250.0)
    t = 0.0
    for seq in range(120):
        t = seq / true_sps
        if seq in (40, 41, 90):
            continue  # simulate dropped BLE notifications
        tracker.update(seq & 0xFF, timestamp=t)

    assert abs(tracker.estimated_sps - true_sps) < 0.5
