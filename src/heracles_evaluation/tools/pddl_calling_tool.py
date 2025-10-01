import os

from heracles_evaluation.tool_interface import FunctionParameter, ToolDescription
from heracles_evaluation.tool_registry import ToolRegistry, register_tool


def send_pddl(pddl_goal_string, robot_name: str = None, planner_topic: str = None):
    if robot_name is None or planner_topic is None:
        raise ValueError("send_pddl called with robot_name or planner_topic missing")

    cmd = f"""
    ros2 topic pub {planner_topic} omniplanner_msgs/msg/PddlGoalMsg "{{robot_id: '{robot_name}', pddl_goal: '{pddl_goal_string}'}}" -1
    """
    print("cmd: ", cmd)
    os.system(f"echo '{cmd}' > pddl_test.txt")
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
