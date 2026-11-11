"""Command-line interface for OctoKin (`octokin`).

Subcommands:
  octokin scan    Discover nearby PiEEG Octopus-16 sensors over Bluetooth LE
  octokin check   5-second / live 16-pin dry electrode contact & trigger check
  octokin stream  Stream JSONL human feedback frames to stdout (with optional CSV recording)
  octokin bridge  Forward host BLE (or demo) packets over UDP into Docker / WSL2 / Isaac Sim
  octokin ros2    Launch the ROS 2 publisher & /cmd_vel safety arbitrator node
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from contextlib import ExitStack

from . import __version__
from .feedback import FeedbackConfig, FeedbackEngine, HumanFeedbackFrame
from .sources import (
    BRIDGE_DEFAULT_HOST,
    BRIDGE_DEFAULT_PORT,
    CSVRecorder,
    CSVReplaySource,
    DemoSource,
    Octopus16BLESource,
    UDPBridgeReceiver,
    UDPBridgeSender,
    scan_octopus_devices,
)


def _make_source(args: argparse.Namespace):
    src = args.source
    if src == "ble":
        return Octopus16BLESource(address=getattr(args, "address", None), reconnect=False)
    if src == "demo":
        return DemoSource(realtime=not getattr(args, "fast", False))
    if src == "udp":
        return UDPBridgeReceiver(port=getattr(args, "port", BRIDGE_DEFAULT_PORT))
    if src == "csv":
        if not getattr(args, "csv", None):
            raise SystemExit("error: --csv PATH is required when --source csv")
        return CSVReplaySource(args.csv, realtime=not getattr(args, "fast", False))
    raise SystemExit(f"Unknown source: {src}")


def format_pin_bar(frame: HumanFeedbackFrame) -> str:
    """Render compact 16-pin contact indicator + physiological metrics for terminal check."""
    pins = "".join("●" if ok else "○" for ok in frame.good_pin_mask)
    status = "ARM-READY" if frame.contact_ok else "CHECK-PINS"
    trig = []
    if frame.jaw_clench:
        trig.append("JAW-CLENCH[HALT]")
    if frame.double_blink:
        trig.append("DOUBLE-BLINK")
    trig_str = " | ".join(trig) if trig else "-"
    return (
        f"[{status:10s}] Pins[{pins}] Q={frame.signal_quality * 100:5.1f}% "
        f"SPS={frame.estimated_sps:5.1f} Loss={frame.packet_loss_rate * 100:4.1f}% | "
        f"Alpha={frame.alpha_ratio:4.2f} Speed={frame.speed_scale:4.2f} "
        f"EMG={frame.emg_rms_uv:5.1f}uV | {trig_str}"
    )


async def _cmd_scan(args: argparse.Namespace) -> int:
    print(f"Scanning Bluetooth LE for Octopus-16 sensors ({args.timeout:.1f}s)...", file=sys.stderr)
    devices = await scan_octopus_devices(timeout=args.timeout)
    if not devices:
        print("No Octopus-16 devices found. Ensure power bank is connected (slow LED blink).")
        return 1
    for d in devices:
        rssi = f" (RSSI {d['rssi']} dBm)" if d.get("rssi") is not None else ""
        print(f"Found: {d['name']}  [{d['address']}]{rssi}")
    return 0


async def _cmd_check(args: argparse.Namespace) -> int:
    source = _make_source(args)
    notch = None if args.notch == 0 else float(args.notch)
    engine = FeedbackEngine(FeedbackConfig(notch_hz=notch))

    t_start = time.monotonic()
    last_print = 0.0
    frames_seen = 0
    last_frame: HumanFeedbackFrame | None = None

    print(
        f"OctoKin Contact & Trigger Check (source={args.source}, duration={args.duration}s)",
        file=sys.stderr,
    )
    async for sample in source:
        frame = engine.process(sample)
        last_frame = frame
        frames_seen += 1
        now = time.monotonic()
        if frame.double_blink or (now - last_print) >= 0.20:
            last_print = now
            print(format_pin_bar(frame))
        if args.duration > 0 and (now - t_start) >= args.duration:
            break

    if frames_seen == 0 or last_frame is None:
        print("No samples received.", file=sys.stderr)
        return 2
    return 0 if last_frame.contact_ok else 1


async def _cmd_stream(args: argparse.Namespace) -> int:
    source = _make_source(args)
    notch = None if args.notch == 0 else float(args.notch)
    engine = FeedbackEngine(FeedbackConfig(notch_hz=notch))
    min_dt = 1.0 / max(1.0, float(args.rate))
    last_emit = 0.0
    count = 0

    with ExitStack() as stack:
        recorder = stack.enter_context(CSVRecorder(args.record)) if args.record else None
        async for sample in source:
            if recorder is not None:
                recorder.write(sample)
            frame = engine.process(sample)
            now = time.monotonic()
            if frame.double_blink or (now - last_emit) >= min_dt or args.fast:
                last_emit = now
                print(frame.to_json(), flush=True)
                count += 1
                if args.max_frames and count >= args.max_frames:
                    break
    return 0


async def _cmd_bridge(args: argparse.Namespace) -> int:
    source = (
        DemoSource(realtime=True)
        if args.source == "demo"
        else Octopus16BLESource(address=args.address, reconnect=True)
    )
    print(
        f"OctoKin UDP Bridge: {args.source} -> udp://{args.host}:{args.port}",
        file=sys.stderr,
    )
    sent = 0
    t0 = time.monotonic()
    with UDPBridgeSender(host=args.host, port=args.port) as sender:
        async for sample in source:
            sender.send(sample)
            sent += 1
            if args.max_samples and sent >= args.max_samples:
                break
            if args.duration > 0 and (time.monotonic() - t0) >= args.duration:
                break
    print(f"Forwarded {sent} Octopus-16 packets.", file=sys.stderr)
    return 0


def _add_common_source_args(p: argparse.ArgumentParser, default_source: str = "ble") -> None:
    p.add_argument(
        "--source",
        choices=["ble", "demo", "udp", "csv"],
        default=default_source,
        help=f"Signal source (default: {default_source})",
    )
    p.add_argument("--address", default=None, help="BLE MAC/UUID address for Octopus-16")
    p.add_argument(
        "--port",
        type=int,
        default=BRIDGE_DEFAULT_PORT,
        help=f"UDP port when --source udp (default: {BRIDGE_DEFAULT_PORT})",
    )
    p.add_argument("--csv", default=None, help="CSV recording path when --source csv")
    p.add_argument(
        "--notch",
        type=int,
        choices=[0, 50, 60],
        default=50,
        help="Mains notch filter Hz, or 0 to disable (default: 50)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="octokin",
        description=(
            "OctoKin — Open-source human physiological feedback for robots (PiEEG Octopus-16)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"octokin {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # scan
    p_scan = sub.add_parser("scan", help="Scan Bluetooth LE for Octopus-16 sensors")
    p_scan.add_argument("--timeout", type=float, default=6.0, help="Scan timeout in seconds")

    # check
    p_check = sub.add_parser("check", help="Check 16-pin scalp contact quality & triggers")
    _add_common_source_args(p_check, default_source="ble")
    p_check.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Check duration in seconds, 0 for continuous (default: 5.0)",
    )
    p_check.add_argument("--fast", action="store_true", help="Disable realtime sleep in demo/csv")

    # stream
    p_stream = sub.add_parser("stream", help="Stream JSONL HumanFeedbackFrames to stdout")
    _add_common_source_args(p_stream, default_source="ble")
    p_stream.add_argument("--rate", type=float, default=25.0, help="Output frame rate Hz")
    p_stream.add_argument("--record", default=None, help="Save raw 250 SPS microvolts to CSV")
    p_stream.add_argument("--max-frames", type=int, default=0, help="Stop after N emitted frames")
    p_stream.add_argument("--fast", action="store_true", help="Run demo/csv without sleep")

    # bridge
    p_bridge = sub.add_parser(
        "bridge", help="Forward host Octopus-16 BLE stream over UDP into Docker / WSL2 / Isaac Sim"
    )
    p_bridge.add_argument("--source", choices=["ble", "demo"], default="ble")
    p_bridge.add_argument("--address", default=None, help="BLE MAC/UUID address")
    p_bridge.add_argument("--host", default=BRIDGE_DEFAULT_HOST, help="Target UDP host/IP")
    p_bridge.add_argument("--port", type=int, default=BRIDGE_DEFAULT_PORT, help="Target UDP port")
    p_bridge.add_argument("--duration", type=float, default=0.0, help="Stop after seconds (0=inf)")
    p_bridge.add_argument("--max-samples", type=int, default=0, help="Stop after N packets")

    # ros2
    p_ros2 = sub.add_parser("ros2", help="Run the ROS 2 publisher and /cmd_vel safety node")
    _add_common_source_args(p_ros2, default_source="ble")
    p_ros2.add_argument("--publish-raw", action="store_true", help="Publish 250 Hz /octokin/raw_uv")
    p_ros2.add_argument(
        "--latch-blink-pause",
        action="store_true",
        help="Toggle robot pause latch on voluntary double blink",
    )
    p_ros2.add_argument("--rate", type=float, default=25.0, help="Topic publish rate Hz")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "scan":
            return asyncio.run(_cmd_scan(args))
        if args.command == "check":
            return asyncio.run(_cmd_check(args))
        if args.command == "stream":
            return asyncio.run(_cmd_stream(args))
        if args.command == "bridge":
            return asyncio.run(_cmd_bridge(args))
        if args.command == "ros2":
            from .ros2_node import run_ros2_node

            notch = None if args.notch == 0 else float(args.notch)
            run_ros2_node(
                source_kind=args.source,
                ble_address=args.address,
                udp_port=args.port,
                csv_path=args.csv,
                notch_hz=notch,
                publish_raw=args.publish_raw,
                latch_blink_pause=args.latch_blink_pause,
                publish_hz=args.rate,
            )
            return 0
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
