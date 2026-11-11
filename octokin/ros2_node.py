"""ROS 2 integration and velocity safety arbitrator for OctoKin.

Uses standard ROS 2 message types (`std_msgs`, `geometry_msgs/Twist`) so it connects
directly to NVIDIA Isaac Sim's OmniGraph ROS 2 Bridge, Nav2, MoveIt Servo, and Unitree
robots without compiling custom C++ interfaces.

Topics published:
  - /octokin/raw_uv            (std_msgs/Float32MultiArray) : 16-ch µV @ 250 Hz (optional)
  - /octokin/band_powers       (std_msgs/Float32MultiArray) : [delta, theta, alpha, beta, gamma]
  - /octokin/signal_quality    (std_msgs/Float32)           : 0.0 .. 1.0 contact score
  - /octokin/speed_scale       (std_msgs/Float32)           : 0.0 .. 1.0 safe velocity multiplier
  - /octokin/engagement_index  (std_msgs/Float32)           : beta / (alpha + theta)
  - /octokin/intervention      (std_msgs/Bool)              : True on jaw-clench emergency halt
  - /octokin/double_blink      (std_msgs/Bool)              : Pulse True on voluntary double blink
  - /octokin/feedback_json     (std_msgs/String)            : Full HumanFeedbackFrame JSON
  - /cmd_vel                   (geometry_msgs/Twist)        : Contact- & clench-gated robot Twist

Topics subscribed:
  - /cmd_vel_raw               (geometry_msgs/Twist)        : Upstream teleop / autonomy Twist
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass

from .feedback import FeedbackConfig, FeedbackEngine, HumanFeedbackFrame
from .sources import (
    BRIDGE_DEFAULT_PORT,
    CSVReplaySource,
    DemoSource,
    Octopus16BLESource,
    UDPBridgeReceiver,
)


@dataclass(frozen=True, slots=True)
class TwistCommand:
    """Framework-independent 6-DOF velocity command matching geometry_msgs/Twist."""

    linear_x: float = 0.0
    linear_y: float = 0.0
    linear_z: float = 0.0
    angular_x: float = 0.0
    angular_y: float = 0.0
    angular_z: float = 0.0

    def scaled(self, factor: float) -> TwistCommand:
        f = float(factor)
        return TwistCommand(
            linear_x=self.linear_x * f,
            linear_y=self.linear_y * f,
            linear_z=self.linear_z * f,
            angular_x=self.angular_x * f,
            angular_y=self.angular_y * f,
            angular_z=self.angular_z * f,
        )


class TwistSafetyArbitrator:
    """Deterministic safety layer modulating robot Twist commands from human feedback.

    Enforces three robotics safety invariants:
      1. Watchdog timeout: if biosignal frames stop arriving (> watchdog_timeout_s),
         velocity scales immediately to `stale_speed_scale` (default 0.0).
      2. Jaw-clench intervention: immediate 0.0 velocity halt while clenched.
      3. Optional double-blink pause latch: hands-free toggle between armed and paused.
    """

    def __init__(
        self,
        watchdog_timeout_s: float = 0.50,
        latch_pause_on_double_blink: bool = False,
        stale_speed_scale: float = 0.0,
    ):
        self.watchdog_timeout_s = float(watchdog_timeout_s)
        self.latch_pause_on_double_blink = bool(latch_pause_on_double_blink)
        self.stale_speed_scale = float(stale_speed_scale)
        self._latest_frame: HumanFeedbackFrame | None = None
        self._latest_arrival_monotonic: float | None = None
        self.paused_latch: bool = False

    def update_feedback(
        self, frame: HumanFeedbackFrame, now_monotonic: float | None = None
    ) -> None:
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        self._latest_frame = frame
        self._latest_arrival_monotonic = now
        if self.latch_pause_on_double_blink and frame.double_blink:
            self.paused_latch = not self.paused_latch

    def effective_speed_scale(self, now_monotonic: float | None = None) -> float:
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        if self._latest_frame is None or self._latest_arrival_monotonic is None:
            return self.stale_speed_scale
        if (now - self._latest_arrival_monotonic) > self.watchdog_timeout_s:
            return self.stale_speed_scale
        if self.paused_latch or self._latest_frame.intervention:
            return 0.0
        return float(self._latest_frame.speed_scale)

    def modulate(
        self, raw_cmd: TwistCommand, now_monotonic: float | None = None
    ) -> TwistCommand:
        return raw_cmd.scaled(self.effective_speed_scale(now_monotonic=now_monotonic))


def build_source(
    source_kind: str,
    ble_address: str | None = None,
    udp_port: int = BRIDGE_DEFAULT_PORT,
    csv_path: str | None = None,
):
    """Instantiate a configured SignalSource by name ('ble', 'demo', 'udp', 'csv')."""
    kind = source_kind.lower().strip()
    if kind == "ble":
        return Octopus16BLESource(address=ble_address)
    if kind == "demo":
        return DemoSource(realtime=True)
    if kind == "udp":
        return UDPBridgeReceiver(port=udp_port)
    if kind == "csv":
        if not csv_path:
            raise ValueError("--csv-path is required when --source=csv")
        return CSVReplaySource(csv_path, realtime=True)
    raise ValueError(f"Unknown source '{source_kind}'. Choose from: ble, demo, udp, csv.")


def run_ros2_node(
    source_kind: str = "demo",
    ble_address: str | None = None,
    udp_port: int = BRIDGE_DEFAULT_PORT,
    csv_path: str | None = None,
    notch_hz: float | None = 50.0,
    publish_raw: bool = False,
    latch_blink_pause: bool = False,
    publish_hz: float = 25.0,
) -> None:
    """Start the OctoKin ROS 2 publisher and `/cmd_vel` safety arbitrator node."""
    try:
        import rclpy
        from geometry_msgs.msg import Twist
        from rclpy.node import Node
        from std_msgs.msg import Bool, Float32, Float32MultiArray, String
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 Python libraries (`rclpy`, `std_msgs`, `geometry_msgs`) were not found. "
            "Source your ROS 2 Humble/Jazzy setup.bash (or Isaac Sim ROS 2 bridge env) first."
        ) from exc

    rclpy.init()
    node = Node("octokin_node")

    pub_quality = node.create_publisher(Float32, "/octokin/signal_quality", 10)
    pub_speed = node.create_publisher(Float32, "/octokin/speed_scale", 10)
    pub_engage = node.create_publisher(Float32, "/octokin/engagement_index", 10)
    pub_intervene = node.create_publisher(Bool, "/octokin/intervention", 10)
    pub_blink = node.create_publisher(Bool, "/octokin/double_blink", 10)
    pub_bands = node.create_publisher(Float32MultiArray, "/octokin/band_powers", 10)
    pub_json = node.create_publisher(String, "/octokin/feedback_json", 10)
    pub_raw = (
        node.create_publisher(Float32MultiArray, "/octokin/raw_uv", 20) if publish_raw else None
    )
    pub_cmd_vel = node.create_publisher(Twist, "/cmd_vel", 10)

    arbitrator = TwistSafetyArbitrator(latch_pause_on_double_blink=latch_blink_pause)
    lock = threading.Lock()

    def _on_cmd_vel_raw(msg: Twist) -> None:
        raw = TwistCommand(
            linear_x=msg.linear.x,
            linear_y=msg.linear.y,
            linear_z=msg.linear.z,
            angular_x=msg.angular.x,
            angular_y=msg.angular.y,
            angular_z=msg.angular.z,
        )
        with lock:
            safe = arbitrator.modulate(raw)
        out = Twist()
        out.linear.x = safe.linear_x
        out.linear.y = safe.linear_y
        out.linear.z = safe.linear_z
        out.angular.x = safe.angular_x
        out.angular.y = safe.angular_y
        out.angular.z = safe.angular_z
        pub_cmd_vel.publish(out)

    node.create_subscription(Twist, "/cmd_vel_raw", _on_cmd_vel_raw, 10)

    source = build_source(
        source_kind, ble_address=ble_address, udp_port=udp_port, csv_path=csv_path
    )
    engine = FeedbackEngine(FeedbackConfig(notch_hz=notch_hz))
    stop_event = threading.Event()
    min_pub_dt = 1.0 / max(1.0, float(publish_hz))

    def _acquisition_worker() -> None:
        async def _loop() -> None:
            last_pub = 0.0
            async for sample in source:
                if stop_event.is_set():
                    break
                frame = engine.process(sample)
                with lock:
                    arbitrator.update_feedback(frame)

                if pub_raw is not None:
                    raw_msg = Float32MultiArray()
                    raw_msg.data = [float(v) for v in sample.channels_uv]
                    pub_raw.publish(raw_msg)

                now = time.monotonic()
                # Always publish immediately on discrete events (clench edge or double blink)
                if frame.double_blink or (now - last_pub) >= min_pub_dt:
                    last_pub = now
                    pub_quality.publish(Float32(data=float(frame.signal_quality)))
                    pub_speed.publish(Float32(data=float(frame.speed_scale)))
                    pub_engage.publish(Float32(data=float(frame.engagement_index)))
                    pub_intervene.publish(Bool(data=bool(frame.intervention)))
                    pub_blink.publish(Bool(data=bool(frame.double_blink)))

                    bands_msg = Float32MultiArray()
                    bands_msg.data = [
                        float(frame.band_powers.get(k, 0.0))
                        for k in ("delta", "theta", "alpha", "beta", "gamma")
                    ]
                    pub_bands.publish(bands_msg)
                    pub_json.publish(String(data=frame.to_json()))

        asyncio.run(_loop())

    worker = threading.Thread(target=_acquisition_worker, daemon=True)
    worker.start()
    node.get_logger().info(
        f"OctoKin ROS 2 node started (source={source_kind}, notch={notch_hz} Hz)"
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if hasattr(source, "stop"):
            source.stop()
        worker.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    run_ros2_node()
