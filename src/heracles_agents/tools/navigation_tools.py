import os

from heracles_agents.tool_interface import FunctionParameter, ToolDescription
from heracles_agents.tool_registry import ToolRegistry, register_tool


def _build_action_msg_yaml(action_type, scalar_value=0.0, stand_sit_action=""):
    return (
        f"{{action_type: '{action_type}', "
        f"scalar_value: {scalar_value}, "
        f"stand_sit_action: '{stand_sit_action}'}}"
    )


def _publish_action(robot_name, executor_topic, action_type, scalar_value=0.0, stand_sit_action=""):
    action_yaml = _build_action_msg_yaml(action_type, scalar_value, stand_sit_action)
    msg_yaml = (
        f"{{header: {{stamp: {{sec: 0, nanosec: 0}}, frame_id: ''}}, "
        f"plan_id: 'direct_command', "
        f"robot_name: '{robot_name}', "
        f"actions: [{action_yaml}]}}"
    )
    cmd = (
        f"ros2 topic pub {executor_topic} "
        f"robot_executor_msgs/msg/ActionSequenceMsg \"{msg_yaml}\" -1"
    )
    print("cmd: ", cmd)
    os.system(cmd)


def send_move_relative(distance_m: float, robot_name: str = None, executor_topic: str = None):
    if robot_name is None or executor_topic is None:
        raise ValueError("send_move_relative called with robot_name or executor_topic missing")
    _publish_action(robot_name, executor_topic, "MOVE_RELATIVE", scalar_value=distance_m)
    direction = "forward" if distance_m >= 0 else "backward"
    return f"Sent move_relative({abs(distance_m)}m {direction}) to {robot_name}"


def send_turn_relative(angle_deg: float, robot_name: str = None, executor_topic: str = None):
    if robot_name is None or executor_topic is None:
        raise ValueError("send_turn_relative called with robot_name or executor_topic missing")
    _publish_action(robot_name, executor_topic, "TURN_RELATIVE", scalar_value=angle_deg)
    direction = "left" if angle_deg >= 0 else "right"
    return f"Sent turn_relative({abs(angle_deg)} degrees {direction}) to {robot_name}"


def send_strafe(distance_m: float, robot_name: str = None, executor_topic: str = None):
    if robot_name is None or executor_topic is None:
        raise ValueError("send_strafe called with robot_name or executor_topic missing")
    _publish_action(robot_name, executor_topic, "STRAFE", scalar_value=distance_m)
    direction = "left" if distance_m >= 0 else "right"
    return f"Sent strafe({abs(distance_m)}m {direction}) to {robot_name}"


def send_stop(robot_name: str = None, executor_topic: str = None):
    if robot_name is None or executor_topic is None:
        raise ValueError("send_stop called with robot_name or executor_topic missing")
    _publish_action(robot_name, executor_topic, "STOP")
    return f"Sent stop to {robot_name}"


def send_stand_sit(action: str, robot_name: str = None, executor_topic: str = None):
    if robot_name is None or executor_topic is None:
        raise ValueError("send_stand_sit called with robot_name or executor_topic missing")
    if action not in ("stand", "sit"):
        raise ValueError(f"stand_sit action must be 'stand' or 'sit', got: {action!r}")
    _publish_action(robot_name, executor_topic, "STAND_SIT", stand_sit_action=action)
    return f"Sent {action} to {robot_name}"


# --- Tool registrations ---

move_tool = ToolDescription(
    name="move_relative",
    description="Move the robot forward or backward by a specified distance in meters. Positive values move forward, negative values move backward.",
    parameters=[
        FunctionParameter(
            "distance_m", float,
            "Distance in meters. Positive = forward, negative = backward.",
        ),
    ],
    function=send_move_relative,
)
register_tool(move_tool)

turn_tool = ToolDescription(
    name="turn_relative",
    description="Turn the robot left or right by a specified angle in degrees. Positive values turn left (counter-clockwise), negative values turn right (clockwise).",
    parameters=[
        FunctionParameter(
            "angle_deg", float,
            "Angle in degrees. Positive = left (CCW), negative = right (CW).",
        ),
    ],
    function=send_turn_relative,
)
register_tool(turn_tool)

strafe_tool = ToolDescription(
    name="strafe",
    description="Move the robot sideways by a specified distance in meters. Positive values move left, negative values move right.",
    parameters=[
        FunctionParameter(
            "distance_m", float,
            "Distance in meters. Positive = left, negative = right.",
        ),
    ],
    function=send_strafe,
)
register_tool(strafe_tool)

stop_tool = ToolDescription(
    name="stop_robot",
    description="Immediately stop all robot motion and cancel any in-progress actions.",
    parameters=[],
    function=send_stop,
)
register_tool(stop_tool)

stand_sit_tool = ToolDescription(
    name="stand_sit",
    description="Make the robot stand up or sit down.",
    parameters=[
        FunctionParameter(
            "action", str,
            "Either 'stand' or 'sit'.",
            enum_values=["stand", "sit"],
        ),
    ],
    function=send_stand_sit,
)
register_tool(stand_sit_tool)

print("Registered navigation tools: ")
print(ToolRegistry.registered_tool_summary())
