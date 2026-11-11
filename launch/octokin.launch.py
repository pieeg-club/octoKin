"""ROS 2 launch file for OctoKin (`ros2 launch octokin octokin.launch.py` or standalone).

Examples:
  ros2 launch launch/octokin.launch.py source:=demo
  ros2 launch launch/octokin.launch.py source:=ble notch:=60
  ros2 launch launch/octokin.launch.py source:=udp port:=16160
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    source_arg = DeclareLaunchArgument(
        "source",
        default_value="ble",
        description="Octopus-16 input source: ble, demo, or udp",
    )
    notch_arg = DeclareLaunchArgument(
        "notch",
        default_value="50",
        description="Powerline notch filter frequency in Hz (50, 60, or 0)",
    )
    port_arg = DeclareLaunchArgument(
        "port",
        default_value="16160",
        description="UDP port when source:=udp (forwarded by `octokin bridge`)",
    )

    node_proc = ExecuteProcess(
        cmd=[
            "python3",
            "-m",
            "octokin.cli",
            "ros2",
            "--source",
            LaunchConfiguration("source"),
            "--notch",
            LaunchConfiguration("notch"),
            "--port",
            LaunchConfiguration("port"),
        ],
        output="screen",
    )

    return LaunchDescription([source_arg, notch_arg, port_arg, node_proc])
