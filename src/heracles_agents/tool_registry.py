import inspect


class ToolRegistry:
    tools = {}

    @classmethod
    def registered_tool_summary(cls):
        return list(cls.tools.keys())

    @classmethod
    def get_arg_type(cls, tool_name, arg_name):
        if tool_name not in ToolRegistry.tools:
            raise ValueError(f"Tool {tool_name} is not present in the tool register")
        fn = cls.tools[tool_name].function
        signature = inspect.signature(fn)
        if arg_name not in signature.parameters:
            raise ValueError(f"{arg_name} is not a valid argument name for {tool_name}")
        arg_type = signature.parameters[arg_name].annotation
        if arg_type == inspect._empty:
            raise ValueError(
                f"Tool {tool_name} does not have a type associated \
                with argument {arg_name}, so we cannot instantiate it from yaml"
            )
        return arg_type


def register_tool(tool_description):
    name = tool_description.name
    if name in ToolRegistry.tools:
        print(f"{name} already has registered function!")
    else:
        ToolRegistry.tools[name] = tool_description
