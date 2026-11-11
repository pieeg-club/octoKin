# OctoKin

**Open-source human feedback for robots.**  
*By [PiEEG](https://www.pieeg.com) · Powered by [Octopus-16](https://github.com/pieeg-club/Octopus_16) · Built for ROS 2 and NVIDIA Isaac*

> *"Your robot can see you. Hear you. Now it can respond to your body's signals."*

**OctoKin** connects live EEG, EMG, and EOG biosignals from the coin-sized **PiEEG Octopus-16** dry-electrode sensor directly to **ROS 2**, **NVIDIA Isaac Sim**, and **Reinforcement Learning** (`Gymnasium` / `Isaac Lab`).

OctoKin is built around **measurable physiological human feedback and explicit operator signals** — never unscientific "mind-reading" claims:
- **16-Pin Contact Quality Gate:** Continuously verifies scalp contact on all 16 spring-loaded pogo pins so loose electrodes never cause erratic robot motion.
- **Fast Physiological Triggers (~50 ms):** Voluntary jaw-clench EMG bursts trigger instant `/cmd_vel` safety halts; voluntary double blinks trigger hands-free mode or waypoint switches.
- **Continuous Shared-Autonomy Modulation:** Smooth Welch spectral ratios (`alpha_ratio`, `engagement_index`) scale robot velocity or compliance within safe bounds.
- **RL Human-in-the-Loop Wrapper:** Synchronizes 250 SPS physiological feedback and explicit preference labels (`+1 / -1`) with 50–100 Hz robot policy steps while gating out noisy frames.

---

> [!CAUTION]
> **Safety Notice:** Octopus-16 is **not a medical device** and has **no galvanic isolation** from its power supply. **Power it ONLY from a small 5 V USB power bank (≤ 2 A) while worn.** Never wear Octopus-16 while plugged into a computer or wall charger. Read the [Octopus-16 Safety Guide](https://docs.pieeg.com/hardware/devices/Octopus_16/safety/).

---

## Architecture

```text
  PiEEG Octopus-16 (26 mm, 18 dry pogo pins, 2x ADS131M08 @ 250 SPS)
         │
         ├── Direct BLE 5 (`Octopus-16-XXXX`) ────────────────────────┐
         └── Host `octokin bridge` ──► UDP (:16160) [Docker / WSL2] ──┤
                                                                      ▼
                                                     ┌─────────────────────────────────┐
                                                     │          OctoKin Core           │
                                                     │  • 50/60 Hz IIR Notch           │
                                                     │  • EEG (1–45 Hz) + EMG (35–100) │
                                                     │  • 16-Pin Contact Quality Gate  │
                                                     │  • Clench / Blink / Band Ratios │
                                                     └───────┬─────────────────┬───────┘
                                                             │                 │
                             ┌───────────────────────────────┘                 └──────────────────────────────┐
                             ▼                                                                                ▼
        ┌─────────────────────────────────────────┐                                      ┌─────────────────────────────────────────┐
        │       ROS 2 & NVIDIA Isaac Sim          │                                      │     Gymnasium / NVIDIA Isaac Lab RL     │
        │  • /octokin/signal_quality  (Float32)   │                                      │  • `HumanFeedbackWrapper(env)`          │
        │  • /octokin/intervention    (Bool)      │                                      │  • `info["octokin"]` synchronized state │
        │  • /octokin/speed_scale     (Float32)   │                                      │  • Jaw-clench penalty & preference RLHF │
        │  • /cmd_vel_raw ──► Safe /cmd_vel       │                                      └─────────────────────────────────────────┘
        └─────────────────────────────────────────┘
```

*Why standard `std_msgs` and `geometry_msgs/Twist`?*  
NVIDIA Isaac Sim's built-in `omni.isaac.ros2_bridge`, Nav2, MoveIt Servo, and Unitree ROS 2 stacks speak `Twist`, `Float32`, and `Bool` natively — **zero custom C++ message compilation required**.

---

## Quick Start (30 Seconds — No Hardware Needed)

```bash
git clone https://github.com/pieeg-club/octoKin.git
cd octoKin
pip install -e ".[dev]"

# 1. Run the 16-pin contact & trigger check on synthetic physiological data
octokin check --source demo --duration 5

# 2. Run the shared-autonomy velocity & jaw-clench safety controller demo
python examples/isaac_sim_controller.py --source demo --duration 11 --fast

# 3. Run the RL human-feedback wrapper demo
python examples/rl_human_feedback_demo.py
```

---

## Live Setup with PiEEG Octopus-16

1. **Power** your flashed Octopus-16 (`XIAO ESP32-S3`) from a **5 V battery power bank** (LED blinks slowly).
2. **Place** the 18-pin cluster over **Pz** (or **Cz** / **Fp**), part hair slightly, press straight down, and secure with a strap.
3. **Scan & verify pin contact** before arming your robot:

```bash
pip install -e ".[ble]"

octokin scan
octokin check --source ble --notch 50   # use --notch 60 in 60 Hz regions
```

Wait until `octokin check` reports `[ARM-READY ]` (`Q >= 60%`), then test a brief **jaw clench** to confirm `JAW-CLENCH[HALT]` triggers cleanly.

---

## Using with ROS 2 & NVIDIA Isaac Sim

### Option A: Native Linux Host (Direct BLE)

```bash
source /opt/ros/jazzy/setup.bash   # or humble
octokin ros2 --source ble --notch 50
```

### Option B: Docker / WSL2 / Remote GPU Workstation (`octokin bridge`)

Docker containers and WSL2 often lack direct Bluetooth LE access. Run `octokin bridge` on your host laptop and `octokin ros2 --source udp` inside your ROS 2 / Isaac Sim container:

```bash
# Terminal 1 — Host machine with Bluetooth (Windows, macOS, or Linux)
octokin bridge --source ble --host 127.0.0.1 --port 16160

# Terminal 2 — Inside ROS 2 / NVIDIA Isaac Sim container
octokin ros2 --source udp --port 16160
```

Or launch via `ros2 launch`:

```bash
ros2 launch launch/octokin.launch.py source:=udp port:=16160 notch:=50
```

### ROS 2 Topic Reference

| Topic | Type | Description |
|---|---|---|
| `/octokin/signal_quality` | `std_msgs/Float32` | `0.0–1.0` dry pogo-pin contact & packet integrity score |
| `/octokin/intervention` | `std_msgs/Bool` | `True` during voluntary jaw-clench emergency halt |
| `/octokin/double_blink` | `std_msgs/Bool` | Pulses `True` on voluntary double blink |
| `/octokin/speed_scale` | `std_msgs/Float32` | Contact- & clench-gated velocity multiplier (`0.0–1.0`) |
| `/octokin/engagement_index` | `std_msgs/Float32` | `beta / (alpha + theta)` spectral ratio |
| `/octokin/band_powers` | `std_msgs/Float32MultiArray` | `[delta, theta, alpha, beta, gamma]` in µV²/Hz |
| `/octokin/feedback_json` | `std_msgs/String` | Compact JSON snapshot of `HumanFeedbackFrame` |
| `/cmd_vel_raw` → `/cmd_vel` | `geometry_msgs/Twist` | Scales upstream velocity; halts to `0.0` on clench, bad contact, or 500 ms watchdog timeout |

---

## Using in Reinforcement Learning (Gymnasium / Isaac Lab)

```python
from octokin import HumanFeedbackWrapper, RewardShapingConfig

env = HumanFeedbackWrapper(
    base_env,
    reward_config=RewardShapingConfig(
        intervention_penalty=-1.0,
        explicit_preference_weight=1.0,
        alpha_calm_bonus_weight=0.05,
    ),
)

obs, _ = env.reset()
# Feed 250 SPS OctopusSample frames collected during the policy step:
env.push_samples(latest_octopus_samples)
obs, reward, terminated, truncated, info = env.step(action)

if info["octokin"]["valid_feedback"] and info["octokin"]["intervention"]:
    print("Operator intervened via jaw clench!")
```

---

## Octopus-16 Wire Protocol Summary

| Parameter | Value |
|---|---|
| **BLE Service UUID** | `4fafc201-1fb5-459e-8fcc-c5c9c331914b` |
| **Data Characteristic** | `beb5483e-36e1-4688-b7f5-ea07361b26a8` (Notify) |
| **Packet Format (51 B)** | `0xA0` · `counter (0–255)` · `16 × int24_be` · `0xC0` |
| **Conversion** | `raw * (1.2 / 4.0 / 8388607) * 1e6` ≈ `0.0358 µV/count` |
| **Sample Rate** | `250 SPS` (`8.192 MHz` clock) or `~244.14 SPS` (`8.000 MHz` clock — auto-estimated) |

---

## Development & Testing

```bash
pip install -e ".[dev]"
ruff check .
pytest -v
```

## License

MIT License — see [LICENSE](./LICENSE). Built with care for open neuroscience and robotics by [PiEEG](https://www.pieeg.com).
