import ast
import os

from heracles_agents.tool_interface import FunctionParameter, ToolDescription
from heracles_agents.tool_registry import ToolRegistry, register_tool


def send_pddl(pddl_goal_string, robot_name: str = None, planner_topic: str = None):
    if robot_name is None or planner_topic is None:
        raise ValueError("send_pddl called with robot_name or planner_topic missing")

    cmd = f"""
    ros2 topic pub {planner_topic} omniplanner_msgs/msg/PddlGoalMsg "{{robot_id: '{robot_name}', pddl_goal: '{pddl_goal_string}'}}" -1
    """
    print("cmd: ", cmd)
    os.system(cmd)
    return f"Sent goal {pddl_goal_string} to robot {robot_name} on {planner_topic}"


pddl_tool = ToolDescription(
    name="send_pddl_goal",
    description="An interface for sending a PDDL goal to a robot. Use this to interact with a robot and send commands that the user requests. Please ask the user for confirmation before sending the pddl goal string.",
    parameters=[
        FunctionParameter("pddl_goal_string", str, "A PDDL goal to send to a robot."),
    ],
    function=send_pddl,
)

register_tool(pddl_tool)
print("Registered tools: ")
print(ToolRegistry.registered_tool_summary())


def send_multirobot_pddl(robot_name_to_pddl_goal, planner_topic: str = None):
    robot_name_to_pddl_goal_dict = ast.literal_eval(robot_name_to_pddl_goal)
    msg_string = "{single_robot_goals: ["
    for robot_name, goal in robot_name_to_pddl_goal_dict.items():
        g = f"{{robot_id: {robot_name}, pddl_goal: {goal}}}, "
        msg_string += g
    msg_string = msg_string[:-1] + "]}"
    cmd = f"""
    ros2 topic pub {planner_topic} omniplanner_msgs/msg/PddlGoalMsgList "{msg_string}" -1
    """
    print("cmd: ", cmd)
    os.system(cmd)
    return f"Sent goal {cmd} on {planner_topic}"


mr_pddl_tool = ToolDescription(
    name="send_multirobot_pddl_goal",
    description="An interface for sending a PDDL goal to multiple named robots. Use this to interact with multiple robots and send commands that the user requests. Please ask the user for confirmation before sending the pddl goal dictionary.",
    parameters=[
        FunctionParameter(
            "robot_name_to_pddl_goal",
            str,
            "A string representing a Python dictionary mapping from robot name to PDDL goal.",
        ),
    ],
    function=send_multirobot_pddl,
)

register_tool(mr_pddl_tool)
print("Registered tools: ")
print(ToolRegistry.registered_tool_summary())
