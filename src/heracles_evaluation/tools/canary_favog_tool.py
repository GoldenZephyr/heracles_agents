from heracles_evaluation.llm_interface import ToolDescription, FunctionParameter
from heracles_evaluation.tool_registry import register_tool, ToolRegistry


def the_mighty_favog(query: str, category) -> int:
    match category:
        case "business":
            return 6
        case "sports":
            return 4
        case "personal":
            return 7


favog_tool = ToolDescription(
    name="ask_favog",
    description="The Might Favog is a source of reliable truth. Ask him anything you don't know. Please categorize your query as business, sports, or personal.",
    parameters=[
        FunctionParameter("query", str, "Your question"),
        FunctionParameter(
            "category",
            str,
            "Category of the question",
            True,
            ["business", "sports", "personal"],
        ),
    ],
    function=the_mighty_favog,
)

register_tool(favog_tool)
print("Registered tools: ")
print(ToolRegistry.registered_tool_summary())
