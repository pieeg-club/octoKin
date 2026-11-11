"""Human-in-the-loop Reinforcement Learning wrapper for Gymnasium and NVIDIA Isaac Lab.

Synchronizes live or replayed Octopus-16 physiological streams with environment `step()`
transitions, exposing:
  - Contact-gated physiological state in `info["octokin"]` (corrupted pins never poison buffers)
  - Optional reward shaping combining jaw-clench intervention penalties, explicit preference
    labels (+1 / -1), and smooth physiological comfort/calmness terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .feedback import FeedbackConfig, FeedbackEngine, HumanFeedbackFrame
from .protocol import OctopusSample

try:
    import gymnasium as _gym

    _BaseWrapper = _gym.Wrapper
except ImportError:

    class _BaseWrapper:  # type: ignore[no-redef]
        """Lightweight fallback wrapper when gymnasium is not installed."""

        def __init__(self, env: Any):
            self.env = env

        def __getattr__(self, name: str) -> Any:
            return getattr(self.env, name)


@dataclass(slots=True)
class RewardShapingConfig:
    """Weights for combining task reward with human physiological & explicit feedback."""

    enable_shaping: bool = True
    intervention_penalty: float = -1.0
    explicit_preference_weight: float = 1.0
    alpha_calm_bonus_weight: float = 0.05
    terminate_on_intervention: bool = False


class HumanFeedbackWrapper(_BaseWrapper):
    """Wraps a Gymnasium or Isaac Lab single/vectorized environment with OctoKin feedback."""

    def __init__(
        self,
        env: Any,
        feedback_config: FeedbackConfig | None = None,
        reward_config: RewardShapingConfig | None = None,
    ):
        super().__init__(env)
        self.engine = FeedbackEngine(feedback_config)
        self.reward_config = reward_config or RewardShapingConfig()
        self._latest_frame: HumanFeedbackFrame | None = None
        self._pending_preference: float = 0.0
        self._had_intervention_since_step: bool = False
        self._had_double_blink_since_step: bool = False

    @property
    def latest_feedback(self) -> HumanFeedbackFrame | None:
        return self._latest_frame

    def push_sample(self, sample: OctopusSample) -> HumanFeedbackFrame:
        """Feed one Octopus-16 sample arrived between RL environment steps."""
        frame = self.engine.process(sample)
        self._latest_frame = frame
        if frame.intervention:
            self._had_intervention_since_step = True
        if frame.double_blink:
            self._had_double_blink_since_step = True
        return frame

    def push_samples(self, samples: list[OctopusSample] | tuple[OctopusSample, ...]) -> None:
        """Feed a batch of Octopus-16 samples (e.g. 5 samples per 50 Hz robot step)."""
        for s in samples:
            self.push_sample(s)

    def record_preference(self, score: float) -> None:
        """Record an explicit human preference/comfort rating (e.g. +1.0, -1.0) for the step."""
        self._pending_preference += float(score)

    def reset(self, **kwargs: Any):
        self._pending_preference = 0.0
        self._had_intervention_since_step = False
        self._had_double_blink_since_step = False
        return self.env.reset(**kwargs)

    def step(self, action: Any):
        obs, task_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info) if info is not None else {}

        frame = self._latest_frame
        valid = bool(frame is not None and frame.contact_ok)
        had_clench = self._had_intervention_since_step or (
            frame is not None and frame.intervention
        )
        intervened = bool(valid and had_clench)
        double_blink = bool(valid and self._had_double_blink_since_step)
        preference = self._pending_preference

        # Clear step-latched accumulators
        self._pending_preference = 0.0
        self._had_intervention_since_step = False
        self._had_double_blink_since_step = False

        human_bonus = 0.0
        if self.reward_config.enable_shaping:
            if preference != 0.0:
                human_bonus += self.reward_config.explicit_preference_weight * preference
            if valid and frame is not None:
                if intervened:
                    human_bonus += self.reward_config.intervention_penalty
                human_bonus += self.reward_config.alpha_calm_bonus_weight * frame.alpha_ratio

        if intervened and self.reward_config.terminate_on_intervention:
            truncated = True

        info["octokin"] = {
            "valid_feedback": valid,
            "signal_quality": frame.signal_quality if frame is not None else 0.0,
            "intervention": intervened,
            "double_blink": double_blink,
            "preference": preference,
            "alpha_ratio": frame.alpha_ratio if valid and frame is not None else 0.0,
            "engagement_index": frame.engagement_index if valid and frame is not None else 0.0,
            "speed_scale": frame.speed_scale if frame is not None else 0.0,
            "human_reward_delta": human_bonus,
            "task_reward": float(task_reward),
        }

        total_reward = float(task_reward) + human_bonus
        return obs, total_reward, terminated, truncated, info
