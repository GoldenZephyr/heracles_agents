"""PDDL goal + constraints tool for execution-time plan repair.

Publishes a ConstrainedPddlGoalMsg to the goal_manager's input topic so the
omniplanner pipeline can ground the goal with runtime constraints applied.
"""

import json
import subprocess

from heracles_agents.tool_interface import FunctionParameter, ToolDescription
from heracles_agents.tool_registry import register_tool


def _constraints_from_json(constraints_json: str) -> str:
    """Turn a JSON list like [["forbidden-poi","o1"],["forbidden-edge","p1","p2"]]
    into a YAML fragment for ros2 topic pub:

        [{predicate: 'forbidden-poi', symbols: ['o1']},
         {predicate: 'forbidden-edge', symbols: ['p1','p2']}]
    """
    if not constraints_json:
        return "[]"
    try:
        facts = json.loads(constraints_json)
    except Exception as exc:
        raise ValueError(f"constraints_json is not valid JSON: {exc}") from exc

    yaml_parts = []
    for fact in facts:
        if not isinstance(fact, (list, tuple)) or len(fact) < 1:
            continue
        predicate = fact[0]
        symbols = [str(s) for s in fact[1:]]
        symbols_yaml = "[" + ",".join(f"'{s}'" for s in symbols) + "]"
        yaml_parts.append(f"{{predicate: '{predicate}', symbols: {symbols_yaml}}}")
    return "[" + ",".join(yaml_parts) + "]"


def send_pddl_with_constraints(
    pddl_goal_string: str,
    constraints_json: str = "",
    robot_name: str = None,
    planner_topic: str = None,
):
    if robot_name is None or planner_topic is None:
        raise ValueError(
            "send_pddl_with_constraints called with robot_name or planner_topic missing"
        )

    constraints_yaml = _constraints_from_json(constraints_json)
    msg_yaml = (
        f"{{goal: {{robot_id: '{robot_name}', pddl_goal: '{pddl_goal_string}'}}, "
        f"constraints: {constraints_yaml}}}"
    )

    cmd = [
        "ros2",
        "topic",
        "pub",
        planner_topic,
        "omniplanner_msgs/msg/ConstrainedPddlGoalMsg",
        msg_yaml,
        "-1",
    ]
    print(f"[pddl_repair_tool] cmd: {' '.join(cmd)}")
    subprocess.run(cmd)

    result = f"Sent goal {pddl_goal_string} to robot {robot_name}"
    if constraints_json:
        result += f" with constraints {constraints_json}"
    return result


pddl_repair_tool = ToolDescription(
    name="send_pddl_goal_with_constraints",
    description=(
        "Send a PDDL goal to a robot, optionally with runtime constraints. "
        "Use the constraints parameter to forbid the robot from visiting "
        "specific locations or traversing specific edges. Constraints are "
        "persistent — they accumulate across calls until cleared. "
        "Please ask the user for confirmation before sending."
    ),
    parameters=[
        FunctionParameter(
            "pddl_goal_string", str, "A PDDL goal string, e.g. '(visited-place p23778)'"
        ),
        FunctionParameter(
            "constraints_json",
            str,
            (
                "JSON list of constraint facts. Each fact is a list like "
                '[["forbidden-poi", "o188"]] to forbid a location, or '
                '[["forbidden-edge", "p1", "p2"]] to forbid an edge. '
                "Empty string means no new constraints."
            ),
            required=False,
        ),
    ],
    function=send_pddl_with_constraints,
)

register_tool(pddl_repair_tool)
