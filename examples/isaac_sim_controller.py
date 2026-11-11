"""End-to-end shared-autonomy velocity & safety controller demo for NVIDIA Isaac Sim / ROS 2.

Demonstrates the three core OctoKin control primitives on a simulated mobile robot / arm base:
  1. 16-pin dry electrode contact gate (`contact_ok`) before allowing motion
  2. Continuous physiological velocity scaling (`speed_scale`)
  3. Instant 0.0 m/s emergency safety halt on voluntary jaw clench (`intervention`)
     and waypoint advance on voluntary double blink (`double_blink`).

Run standalone without hardware (12-second full physiological cycle):
  python examples/isaac_sim_controller.py --source demo --duration 12

Run live with an Octopus-16 sensor:
  python examples/isaac_sim_controller.py --source ble
"""

from __future__ import annotations

import argparse
import asyncio
import time

from octokin.feedback import FeedbackConfig, FeedbackEngine
from octokin.ros2_node import TwistCommand, TwistSafetyArbitrator, build_source


async def run_demo(source_kind: str, duration_s: float, fast: bool) -> None:
    if source_kind == "demo" and fast:
        from octokin.sources import DemoSource

        source = DemoSource(realtime=False, max_samples=int(duration_s * 250))
    else:
        source = build_source(source_kind)

    engine = FeedbackEngine(FeedbackConfig(notch_hz=50.0))
    arbitrator = TwistSafetyArbitrator(latch_pause_on_double_blink=False)

    nominal_cmd = TwistCommand(linear_x=0.80, angular_z=0.15)
    waypoint_idx = 1
    t_start = time.monotonic()
    last_log = -1.0
    prev_intervention = False

    print("Time(s) | Contact | State          | Raw vx -> Safe vx | Event")
    print("-" * 74)

    async for sample in source:
        frame = engine.process(sample)
        sim_t = frame.sample_index / 250.0
        arbitrator.update_feedback(frame, now_monotonic=sim_t)
        safe_cmd = arbitrator.modulate(nominal_cmd, now_monotonic=sim_t)

        event = ""
        if frame.double_blink:
            waypoint_idx += 1
            event = f"DOUBLE-BLINK -> Waypoint #{waypoint_idx}"
        elif frame.intervention and not prev_intervention:
            event = "JAW-CLENCH -> SAFETY HALT"
        elif prev_intervention and not frame.intervention:
            event = "CLENCH RELEASED -> RESUME"
        prev_intervention = frame.intervention

        if event or (sim_t - last_log) >= 0.50:
            last_log = sim_t
            state = (
                "HALTED"
                if frame.intervention
                else ("DISCONNECTED" if not frame.contact_ok else f"Alpha={frame.alpha_ratio:.2f}")
            )
            print(
                f"{sim_t:6.2f}s | {frame.signal_quality * 100:5.1f}%  | "
                f"{state:14s} | {nominal_cmd.linear_x:4.2f} -> {safe_cmd.linear_x:4.2f} m/s "
                f"| {event}"
            )

        if not fast and duration_s > 0 and (time.monotonic() - t_start) >= duration_s:
            break
        if fast and duration_s > 0 and sim_t >= duration_s:
            break


def main() -> None:
    p = argparse.ArgumentParser(description="OctoKin shared-autonomy velocity & safety demo")
    p.add_argument("--source", choices=["demo", "ble", "udp"], default="demo")
    p.add_argument("--duration", type=float, default=11.0)
    p.add_argument(
        "--fast", action="store_true", help="Run synthetic simulation faster than real time"
    )
    args = p.parse_args()
    asyncio.run(run_demo(args.source, args.duration, args.fast))


if __name__ == "__main__":
    main()
