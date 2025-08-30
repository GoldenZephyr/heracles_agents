from heracles_evaluation.structured_tool_interface import StructuredToolDescription
from heracles_evaluation.tool_registry import ToolRegistry, register_tool
from sldp.lark_parser import get_sldp_lark_grammar

sldp_tool = StructuredToolDescription(
    name="sldp_answer_tool",
    description="Use this tool to submit your final SLDP-formatted answer.",
    grammar=get_sldp_lark_grammar(),
)

register_tool(sldp_tool)
print("Registered tools: ")
print(ToolRegistry.registered_tool_summary())
