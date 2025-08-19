from plum import parametric
from heracles_evaluation.pydantic_discriminated_dispatch import (
    discriminated_union_dispatch,
)
from heracles_evaluation.model_client_interfaces import get_client_union_type
from pydantic import BaseModel, Field, field_validator, field_serializer
from heracles_evaluation.prompt import PromptSettings
from typing import Optional
from heracles_evaluation.tool_registry import ToolRegistry
from heracles_evaluation.tool_interface import ToolDescription
from functools import partial
import copy


class ModelInfo(BaseModel):
    """Settings that affect fundamental model performance.

    e.g., model size, temperature, seed.
    Tool calling details are handled elsewhere
    """

    model: str
    temperature: float
    seed: Optional[int] = None


def apply_bound_args(tool_name, bound_args):
    args_to_bind = {}
    for arg_name, fields in bound_args.items():
        arg_type = ToolRegistry.get_arg_type(tool_name, arg_name)
        arg_instance = arg_type(**fields)
        args_to_bind[arg_name] = arg_instance
    function = partial(ToolRegistry.tools[tool_name].function, **args_to_bind)
    return function


class AgentInfo(BaseModel):
    """Configuration for "agentic" behaviors, e.g., tool calling"""

    prompt_settings: PromptSettings
    tools: dict[str, ToolDescription]
    tool_interface: str  # Openai vs. custom vs. ???
    max_iterations: int

    @field_validator("tools", mode="before")
    @classmethod
    def lookup_tools(cls, tools):
        tool_descriptions = {}
        for t in tools:
            tool_name = t["name"]
            if tool_name not in ToolRegistry.tools:
                raise ValueError(
                    f"Unknown tool {tool_name}. Known tools: {list(  ToolRegistry.tools.keys())}"
                )
            if "bound_args" in t:
                function = apply_bound_args(tool_name, t["bound_args"])
                resolved_tool = copy.deepcopy(ToolRegistry.tools[tool_name])
                resolved_tool.function = function
            else:
                resolved_tool = ToolRegistry.tools[tool_name]
            tool_descriptions[tool_name] = resolved_tool

        return tool_descriptions

    @field_serializer("tools")
    def serialize_tools(self, tools):
        return [tool.name for tool in tools]


model_interface_config_type = get_client_union_type()


@parametric
@discriminated_union_dispatch("client")
class LlmAgent[T](BaseModel):
    agent_info: AgentInfo
    model_info: ModelInfo
    client: model_interface_config_type = Field(discriminator="client_type")
