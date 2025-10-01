from heracles_agents.tool_interface import FunctionParameter, ToolDescription
from heracles_agents.tool_registry import register_tool


def answer_tool(answer: str) -> str:
    """Tool to provide final answers."""
    return f"Final answer: {answer}"


answer_tool_desc = ToolDescription(
    name="answer",
    description="Provide the final answer",
    parameters=[
        FunctionParameter("answer", str, "The final answer"),
    ],
    function=answer_tool,
)

register_tool(answer_tool_desc)
