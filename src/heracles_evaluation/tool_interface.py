from dataclasses import dataclass
from typing import Optional, Any, Callable
from heracles_evaluation.tool_registry import ToolRegistry


def type_to_string(typ):
    match typ():
        case str():
            return "string"
        case float():
            return "float"
        case int():
            return "int"
        case dict():
            return "dict"
        case set():
            return "set"
        case list():
            return "list"


@dataclass
class FunctionParameter:
    """Description of a single parameter for a tool/function call"""

    name: str
    param_type: type
    param_description: str
    required: bool = True
    enum_values: Optional[Any] = None

    def to_openai_responses(self):
        d = {
            self.name: {
                "type": type_to_string(self.param_type),
                "description": self.param_description,
            }
        }
        if self.enum_values is not None:
            d[self.name]["enum"] = self.enum_values
        return d


@dataclass
class ToolDescription:
    """Description of a tool / function"""

    name: str
    description: str
    parameters: list[FunctionParameter]
    function: Callable

    def get_tool_function(self):
        try:
            fn = ToolRegistry.tools[self.name]
        except IndexError as ex:
            print(ex)
            print(
                f"Tool {self.name} not registered in ToolRegistry! Registered tools are {ToolRegistry.registered_tool_summary()}"
            )
        return fn

    def to_openai_responses(self):
        parameter_properties = {}
        for p in self.parameters:
            parameter_properties |= p.to_openai_responses()

        required = [p.name for p in self.parameters if p.required]

        parameter_descriptions = {
            "type": "object",
            "properties": parameter_properties,
            "required": required,
            "additionalProperties": False,
        }

        t = {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": parameter_descriptions,
        }
        print("tool formatted: ")
        print(t)
        return t

    def to_custom(self):
        raise NotImplementedError()
