"""OctoKin — Open-source human physiological feedback for robots.

Powered by PiEEG Octopus-16 · Built for ROS 2, NVIDIA Isaac Sim, and RL.
"""

from .dsp import BANDS, RealtimeDSP
from .feedback import FeedbackConfig, FeedbackEngine, HumanFeedbackFrame
from .protocol import (
    COMMAND_CHAR_UUID,
    DATA_CHAR_UUID,
    NOMINAL_SPS,
    NUM_CHANNELS,
    PACKET_SIZE,
    SCALE_UV_PER_COUNT,
    SERVICE_UUID,
    OctopusSample,
    PacketTracker,
    decode_packet,
    encode_packet,
)
from .rl import HumanFeedbackWrapper, RewardShapingConfig
from .sources import (
    CSVRecorder,
    CSVReplaySource,
    DemoSource,
    Octopus16BLESource,
    UDPBridgeReceiver,
    UDPBridgeSender,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "BANDS",
    "COMMAND_CHAR_UUID",
    "CSVRecorder",
    "CSVReplaySource",
    "DATA_CHAR_UUID",
    "DemoSource",
    "FeedbackConfig",
    "FeedbackEngine",
    "HumanFeedbackFrame",
    "HumanFeedbackWrapper",
    "NOMINAL_SPS",
    "NUM_CHANNELS",
    "Octopus16BLESource",
    "OctopusSample",
    "PACKET_SIZE",
    "PacketTracker",
    "RealtimeDSP",
    "RewardShapingConfig",
    "SCALE_UV_PER_COUNT",
    "SERVICE_UUID",
    "UDPBridgeReceiver",
    "UDPBridgeSender",
    "decode_packet",
    "encode_packet",
]
