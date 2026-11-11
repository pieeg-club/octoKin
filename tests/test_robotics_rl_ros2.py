"""Tests for ROS 2 Twist safety arbitrator, RL HumanFeedbackWrapper, and CLI commands."""

from __future__ import annotations

from pathlib import Path

import pytest

from octokin.cli import main as cli_main
from octokin.feedback import HumanFeedbackFrame
from octokin.rl import HumanFeedbackWrapper, RewardShapingConfig
from octokin.ros2_node import TwistCommand, TwistSafetyArbitrator
from octokin.sources import DemoSource


def _frame(
    *,
    contact_ok: bool = True,
    intervention: bool = False,
    double_blink: bool = False,
    speed_scale: float = 0.8,
) -> HumanFeedbackFrame:
    return HumanFeedbackFrame(
        timestamp=10.0,
        sample_index=250,
        estimated_sps=250.0,
        packet_loss_rate=0.0,
        signal_quality=1.0 if contact_ok else 0.2,
        contact_ok=contact_ok,
        good_pin_mask=tuple([contact_ok] * 16),
        intervention=intervention,
        jaw_clench=intervention,
        double_blink=double_blink,
        emg_rms_uv=45.0 if intervention else 4.0,
        alpha_ratio=0.4,
        engagement_index=0.6,
        speed_scale=0.0 if (intervention or not contact_ok) else speed_scale,
        band_powers={"delta": 1.0, "theta": 2.0, "alpha": 8.0, "beta": 4.0, "gamma": 1.0},
    )


def test_twist_arbitrator_watchdog_and_blink_latch() -> None:
    arb = TwistSafetyArbitrator(watchdog_timeout_s=0.50, latch_pause_on_double_blink=True)
    raw = TwistCommand(linear_x=1.0, angular_z=0.5)

    # Before first frame -> halted (0.0)
    assert arb.modulate(raw, now_monotonic=1.0) == TwistCommand()

    # Valid frame -> scaled by 0.8
    arb.update_feedback(_frame(speed_scale=0.8), now_monotonic=1.0)
    modulated = arb.modulate(raw, now_monotonic=1.2)
    assert pytest.approx(modulated.linear_x) == 0.8
    assert pytest.approx(modulated.angular_z) == 0.4

    # Watchdog stale (> 0.50s without frame) -> halted
    assert arb.modulate(raw, now_monotonic=1.6) == TwistCommand()

    # Double blink latches pause on, second double blink unlatches
    arb.update_feedback(_frame(double_blink=True, speed_scale=0.8), now_monotonic=2.0)
    assert arb.paused_latch is True
    assert arb.modulate(raw, now_monotonic=2.1) == TwistCommand()

    arb.update_feedback(_frame(double_blink=True, speed_scale=0.8), now_monotonic=2.6)
    assert arb.paused_latch is False
    assert pytest.approx(arb.modulate(raw, now_monotonic=2.7).linear_x) == 0.8


class _DummyEnv:
    def reset(self, **_kwargs):
        return [0.0], {"env": "dummy"}

    def step(self, action):
        return [float(action)], 1.0, False, False, {}


def test_rl_wrapper_shapes_rewards_and_gates_bad_contact() -> None:
    env = HumanFeedbackWrapper(
        _DummyEnv(),
        reward_config=RewardShapingConfig(
            intervention_penalty=-2.0,
            explicit_preference_weight=1.5,
            alpha_calm_bonus_weight=0.1,
            terminate_on_intervention=True,
        ),
    )
    env.reset()

    # 1) Disconnected sensor (14 railed pins): physiological shaping must be suppressed!
    bad_synth = DemoSource(realtime=False, disconnected_pins=tuple(range(14)))
    for i in range(40):
        env.push_sample(bad_synth.generate_sample(i))
    env.record_preference(-1.0)
    _, reward, _, truncated, info = env.step(0.5)
    assert info["octokin"]["valid_feedback"] is False
    assert truncated is False
    assert pytest.approx(reward) == 1.0 - 1.5  # task_reward + explicit preference only

    # 2) Clean contact + jaw-clench burst at t=7.2s -> intervention penalty + truncation
    good_synth = DemoSource(realtime=False, scripted_events=True)
    env.reset()
    # Warm up clean contact then step into clench window (samples 1720..1820 -> 6.88s..7.28s)
    for i in range(1600, 1820):
        env.push_sample(good_synth.generate_sample(i))
    _, reward_clench, _, truncated_clench, info_clench = env.step(0.5)
    assert info_clench["octokin"]["valid_feedback"] is True
    assert info_clench["octokin"]["intervention"] is True
    assert truncated_clench is True
    assert reward_clench < 0.0

    # 3) Resetting after intervention must NOT leak stale clench truncation into new episode
    env.reset()
    _, reward_after_reset, _, truncated_after_reset, info_after_reset = env.step(0.5)
    assert info_after_reset["octokin"]["intervention"] is False
    assert truncated_after_reset is False
    assert pytest.approx(reward_after_reset) == 1.0


def test_cli_check_and_stream_record_replay(tmp_path: Path, capsys) -> None:
    csv_path = tmp_path / "cli_rec.csv"

    rc_stream = cli_main(
        [
            "stream",
            "--source",
            "demo",
            "--max-frames",
            "20",
            "--record",
            str(csv_path),
            "--fast",
        ]
    )
    assert rc_stream == 0
    out_lines = [ln for ln in capsys.readouterr().out.strip().splitlines() if ln.strip()]
    assert len(out_lines) == 20
    assert csv_path.exists()

    rc_check = cli_main(["check", "--source", "demo", "--duration", "0.2", "--fast"])
    assert rc_check == 0
