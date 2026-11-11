"""Minimal Human-in-the-Loop Reinforcement Learning example using OctoKin.

Shows how a 50 Hz robot control environment (e.g. NVIDIA Isaac Lab or Gymnasium)
consumes 5 Octopus-16 samples (250 SPS) per RL step with:
  - Contact-gated physiological state in `info["octokin"]`
  - Automatic jaw-clench intervention penalties
  - Explicit human preference reward shaping
"""

from __future__ import annotations

from octokin import DemoSource, HumanFeedbackWrapper, RewardShapingConfig


class MinimalRobotEnv:
    """1-D point-mass robot reaching env compatible with Gymnasium's reset/step signature."""

    def __init__(self) -> None:
        self.pos = 0.0
        self.goal = 4.5
        self.steps = 0

    def reset(self, **_kwargs):
        self.pos = 0.0
        self.steps = 0
        return [self.pos], {}

    def step(self, action: float):
        self.pos += float(action) * 0.02
        self.steps += 1
        task_reward = -abs(self.goal - self.pos)
        terminated = abs(self.goal - self.pos) < 0.02
        truncated = self.steps >= 600
        return [self.pos], task_reward, terminated, truncated, {}


def main() -> None:
    base_env = MinimalRobotEnv()
    env = HumanFeedbackWrapper(
        base_env,
        reward_config=RewardShapingConfig(
            intervention_penalty=-1.5,
            explicit_preference_weight=1.0,
            alpha_calm_bonus_weight=0.1,
        ),
    )

    synth = DemoSource(realtime=False, scripted_events=True)
    obs, _ = env.reset()
    sample_idx = 0

    interventions = 0
    for step_idx in range(450):  # 9 seconds @ 50 Hz robot control
        # 250 SPS biosignal / 50 Hz robot policy = 5 Octopus-16 samples per RL step
        batch = [synth.generate_sample(sample_idx + k) for k in range(5)]
        sample_idx += 5
        env.push_samples(batch)

        if step_idx == 150:
            env.record_preference(+1.0)  # Explicit positive operator rating

        obs, reward, terminated, truncated, info = env.step(action=0.5)
        ok_info = info["octokin"]
        if ok_info["intervention"]:
            interventions += 1

        if step_idx in (120, 150, 360):
            print(
                f"Step {step_idx:3d} | valid={ok_info['valid_feedback']} "
                f"intervene={ok_info['intervention']} pref={ok_info['preference']:+.1f} "
                f"alpha={ok_info['alpha_ratio']:.2f} reward={reward:+.3f}"
            )
        if terminated or truncated:
            break

    print(f"Episode finished cleanly. Total jaw-clench intervention steps: {interventions}")


if __name__ == "__main__":
    main()
