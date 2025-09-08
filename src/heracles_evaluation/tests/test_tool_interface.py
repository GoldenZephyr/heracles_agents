"""
Unit tests for tool_interface module, specifically testing type_to_string function
and FunctionParameter class with all supported types.
"""

import pytest

from heracles_evaluation.tool_interface import (
    FunctionParameter,
    ToolDescription,
)


class TestFunctionParameter:
    """Test the FunctionParameter class with different parameter types."""

    def test_string_parameter(self):
        """Test FunctionParameter with string type."""
        param = FunctionParameter("name", str, "A string parameter")

        # Test serialization methods
        openai_result = param.to_openai_responses()
        expected = {"name": {"type": "string", "description": "A string parameter"}}
        assert openai_result == expected

        # Test that all serialization methods return the same result for basic types
        assert param.to_anthropic() == openai_result
        assert param.to_ollama() == openai_result

    def test_float_parameter(self):
        """Test FunctionParameter with float type."""
        param = FunctionParameter("value", float, "A float parameter")

        openai_result = param.to_openai_responses()
        expected = {"value": {"type": "number", "description": "A float parameter"}}
        assert openai_result == expected

    def test_int_parameter(self):
        """Test FunctionParameter with int type."""
        param = FunctionParameter("count", int, "An integer parameter")

        openai_result = param.to_openai_responses()
        expected = {"count": {"type": "integer", "description": "An integer parameter"}}
        assert openai_result == expected

    def test_dict_parameter(self):
        """Test FunctionParameter with dict type."""
        param = FunctionParameter("config", dict, "A dictionary parameter")

        openai_result = param.to_openai_responses()
        expected = {
            "config": {"type": "object", "description": "A dictionary parameter"}
        }
        assert openai_result == expected

    def test_set_parameter(self):
        """Test FunctionParameter with set type."""
        param = FunctionParameter("tags", set, "A set parameter")

        openai_result = param.to_openai_responses()
        expected = {"tags": {"type": "array", "description": "A set parameter"}}
        assert openai_result == expected

    def test_list_parameter(self):
        """Test FunctionParameter with list type."""
        param = FunctionParameter("items", list, "A list parameter")

        openai_result = param.to_openai_responses()
        expected = {"items": {"type": "array", "description": "A list parameter"}}
        assert openai_result == expected

    def test_parameter_with_enum_values(self):
        """Test FunctionParameter with enum values."""
        param = FunctionParameter(
            "operation",
            str,
            "The operation to perform",
            True,
            ["add", "subtract", "multiply", "divide"],
        )

        openai_result = param.to_openai_responses()
        expected = {
            "operation": {
                "type": "string",
                "description": "The operation to perform",
                "enum": ["add", "subtract", "multiply", "divide"],
            }
        }
        assert openai_result == expected

    def test_optional_parameter(self):
        """Test FunctionParameter with required=False."""
        param = FunctionParameter("optional_param", str, "An optional parameter", False)

        assert param.required is False

        openai_result = param.to_openai_responses()
        expected = {
            "optional_param": {"type": "string", "description": "An optional parameter"}
        }
        assert openai_result == expected

    def test_custom_serialization(self):
        """Test the to_custom method."""
        param = FunctionParameter("test_param", int, "A test parameter")

        custom_result = param.to_custom()
        expected = [
            "Param name: test_param",
            "Param description: A test parameter",
            "type: integer",
        ]
        assert custom_result == expected

    def test_custom_serialization_with_enum(self):
        """Test the to_custom method with enum values."""
        param = FunctionParameter(
            "status", str, "The status", True, ["active", "inactive", "pending"]
        )

        custom_result = param.to_custom()
        expected = [
            "Param name: status",
            "Param description: The status",
            "type: string",
            "Allowed values: ['active', 'inactive', 'pending']",
        ]
        assert custom_result == expected

    @pytest.mark.parametrize(
        "param_type,expected_type_string",
        [
            (str, "string"),
            (float, "number"),
            (int, "integer"),
            (dict, "object"),
            (set, "array"),
            (list, "array"),
        ],
    )
    def test_all_parameter_types(self, param_type, expected_type_string):
        """Parametrized test for FunctionParameter with all supported types."""
        param = FunctionParameter(
            "test_param", param_type, f"A {param_type.__name__} parameter"
        )

        openai_result = param.to_openai_responses()
        assert openai_result["test_param"]["type"] == expected_type_string


class TestToolDescriptionWithAllTypes:
    """Test ToolDescription with tools that use all parameter types."""

    def test_tool_with_all_parameter_types(self):
        """Test a tool that uses all supported parameter types."""

        def multi_type_tool(
            name: str, count: int, value: float, config: dict, tags: set, items: list
        ) -> str:
            """A tool that accepts all parameter types."""
            return f"Processed: {name}, {count}, {value}, {config}, {tags}, {items}"

        tool = ToolDescription(
            name="multi_type_tool",
            description="A tool that accepts all parameter types",
            parameters=[
                FunctionParameter("name", str, "String parameter"),
                FunctionParameter("count", int, "Integer parameter"),
                FunctionParameter("value", float, "Float parameter"),
                FunctionParameter("config", dict, "Dictionary parameter"),
                FunctionParameter("tags", set, "Set parameter"),
                FunctionParameter("items", list, "List parameter"),
            ],
            function=multi_type_tool,
        )

        # Test OpenAI serialization
        openai_result = tool.to_openai_responses()

        # Verify the structure
        assert openai_result["type"] == "function"
        assert openai_result["name"] == "multi_type_tool"
        assert openai_result["description"] == "A tool that accepts all parameter types"

        properties = openai_result["parameters"]["properties"]
        assert properties["name"]["type"] == "string"
        assert properties["count"]["type"] == "integer"
        assert properties["value"]["type"] == "number"
        assert properties["config"]["type"] == "object"
        assert properties["tags"]["type"] == "array"
        assert properties["items"]["type"] == "array"

        # All parameters should be required
        required = openai_result["parameters"]["required"]
        assert set(required) == {"name", "count", "value", "config", "tags", "items"}

        # Test Anthropic serialization
        anthropic_result = tool.to_anthropic()
        assert anthropic_result["name"] == "multi_type_tool"
        assert (
            anthropic_result["description"] == "A tool that accepts all parameter types"
        )

        anthropic_properties = anthropic_result["input_schema"]["properties"]
        assert anthropic_properties["name"]["type"] == "string"
        assert anthropic_properties["count"]["type"] == "integer"
        assert anthropic_properties["value"]["type"] == "number"
        assert anthropic_properties["config"]["type"] == "object"
        assert anthropic_properties["tags"]["type"] == "array"
        assert anthropic_properties["items"]["type"] == "array"

        # Test Ollama serialization
        ollama_result = tool.to_ollama()
        assert ollama_result["type"] == "function"
        assert ollama_result["function"]["name"] == "multi_type_tool"
        assert (
            ollama_result["function"]["description"]
            == "A tool that accepts all parameter types"
        )

        ollama_properties = ollama_result["function"]["parameters"]["properties"]
        assert ollama_properties["name"]["type"] == "string"
        assert ollama_properties["count"]["type"] == "integer"
        assert ollama_properties["value"]["type"] == "number"
        assert ollama_properties["config"]["type"] == "object"
        assert ollama_properties["tags"]["type"] == "array"
        assert ollama_properties["items"]["type"] == "array"

        # Test custom serialization
        custom_result = tool.to_custom()
        assert "Function name: multi_type_tool" in custom_result
        assert (
            "Function Description: A tool that accepts all parameter types"
            in custom_result
        )
        assert "type: string" in custom_result
        assert "type: integer" in custom_result
        assert "type: number" in custom_result
        assert "type: object" in custom_result
        assert "type: array" in custom_result

    def test_tool_with_mixed_required_optional_parameters(self):
        """Test a tool with both required and optional parameters of different types."""

        def mixed_tool(required_str: str, optional_int: int = 42) -> str:
            """A tool with mixed parameter requirements."""
            return f"Required: {required_str}, Optional: {optional_int}"

        tool = ToolDescription(
            name="mixed_tool",
            description="A tool with mixed parameter requirements",
            parameters=[
                FunctionParameter(
                    "required_str", str, "Required string parameter", True
                ),
                FunctionParameter(
                    "optional_int", int, "Optional integer parameter", False
                ),
            ],
            function=mixed_tool,
        )

        openai_result = tool.to_openai_responses()

        # Only required_str should be in the required list
        required = openai_result["parameters"]["required"]
        assert required == ["required_str"]

        # Both parameters should be in properties
        properties = openai_result["parameters"]["properties"]
        assert "required_str" in properties
        assert "optional_int" in properties
        assert properties["required_str"]["type"] == "string"
        assert properties["optional_int"]["type"] == "integer"
