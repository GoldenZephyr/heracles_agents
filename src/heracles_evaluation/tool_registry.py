class ToolRegistry:
    tools = {}

    @classmethod
    def registered_tool_summary(cls):
        return list(cls.tools.keys())


def register_tool(tool_description):
    name = tool_description.name
    if name in ToolRegistry.tools:
        print(f"{name} already has registered function!")
    else:
        ToolRegistry.tools[name] = tool_description
