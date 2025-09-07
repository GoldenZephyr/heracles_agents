from heracles_evaluation.tool_interface import FunctionParameter, ToolDescription
from heracles_evaluation.tool_registry import register_tool


def test_calculator(a: float, b: float, operation: str = "add") -> float:
    """A simple calculator tool for testing."""
    match operation:
        case "add":
            return a + b
        case "subtract":
            return a - b
        case "multiply":
            return a * b
        case "divide":
            return a / b if b != 0 else float("inf")
        case _:
            raise ValueError(f"Unknown operation: {operation}")


calculator_tool = ToolDescription(
    name="calculator",
    description="Perform basic arithmetic operations",
    parameters=[
        FunctionParameter(
            "operation",
            str,
            "The operation to perform",
            True,
            ["add", "subtract", "multiply", "divide"],
        ),
        FunctionParameter("a", float, "First number"),
        FunctionParameter("b", float, "Second number"),
    ],
    function=test_calculator,
)

register_tool(calculator_tool)
