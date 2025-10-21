from heracles_agents.tool_interface import FunctionParameter, ToolDescription
from heracles_agents.tool_registry import ToolRegistry, register_tool
import zmq


def send_penn_cmd(goal, interface_uri: str = None):

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(interface_uri)
    socket.send_string(goal)
    print(f"Sending cmd: {goal} to penn quadrotor")

    return f"Sent goal {goal}"


pddl_tool = ToolDescription(
    name="send_penn_cmd",
    description="An interface for ???",
    parameters=[
        FunctionParameter("goal", str, "A ??? goal ???."),
    ],
    function=send_penn_cmd,
)

register_tool(pddl_tool)
print("Registered tools: ")
print(ToolRegistry.registered_tool_summary())
