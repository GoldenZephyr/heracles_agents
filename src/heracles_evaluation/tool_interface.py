import inspect
from dataclasses import dataclass
from typing import Any, Callable, Optional

from pydantic import BaseModel, PrivateAttr, model_validator

from heracles_evaluation.tool_registry import ToolRegistry


def type_to_string(typ):
    match typ():
        case str():
            return "string"
        case float():
            return "number"
        case int():
            return "integer"
        case dict():
            return "object"
        case set():
            return "array"
        case list():
            return "array"


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

    def to_anthropic(self):
        return self.to_openai_responses()

    def to_custom(self):
        d = [
            f"Param name: {self.name}",
            f"Param description: {self.param_description}",
            f"type: {type_to_string(self.param_type)}",
        ]
        if self.enum_values is not None:
            d.append(f"Allowed values: {str(self.enum_values)}")
        return d


class ToolDescription(BaseModel):
    """Description of a tool / function"""

    name: str
    description: str
    parameters: list[FunctionParameter]
    function: Callable
    _bound_args: PrivateAttr() = None  # dict[str, object]

    def get_tool_function(self):
        try:
            fn = ToolRegistry.tools[self.name]
        except IndexError as ex:
            print(ex)
            print(
                f"Tool {self.name} not registered in ToolRegistry! Registered tools are {ToolRegistry.registered_tool_summary()}"
            )
        return fn

    @model_validator(mode="after")
    def verify_param_names(self):
        function_required_params = set()
        function_optional_params = set()
        for k, v in inspect.signature(self.function).parameters.items():
            if v.default == inspect._empty:
                function_required_params.add(k)
            else:
                function_optional_params.add(k)

        given_param_names = set(p.name for p in self.parameters)

        for gp in given_param_names:
            if gp not in function_optional_params.union(function_required_params):
                raise ValueError(
                    f"Declared parameter {gp} is not an optional or required keyword in defined function"
                )

        if not function_required_params <= given_param_names:
            missing_params = function_required_params - given_param_names
            raise ValueError(
                f"Function requires parameters that are not declared by tool: {missing_params}"
            )

        return self

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
        return t

    def to_anthropic(self):
        parameter_properties = {}
        for p in self.parameters:
            parameter_properties |= p.to_anthropic()

        required = [p.name for p in self.parameters if p.required]
        parameter_descriptions = {
            "type": "object",
            "properties": parameter_properties,
            "required": required,
        }
        t = {
            "name": self.name,
            "description": self.description,
            "input_schema": parameter_descriptions,
        }
        return t

    def to_custom(self):
        parameter_descriptions = ""
        for p in self.parameters:
            d = "\n".join(["  " + s for s in p.to_custom()])
            d = "-" + d[1:]
            parameter_descriptions += d + "\n"
        tool_description = f"""
Function name: {self.name}
Function Description: {self.description}
Parameters:
{parameter_descriptions}
"""
        return tool_description
