from heracles_evaluation.custom_tool_call_parser import lark_parse_tool


def test_simple():
    tool = lark_parse_tool("my_function(a=1, b='cat')")
    assert tool.name == "my_function"
    assert tool.args["a"] == 1
    assert tool.args["b"] == "cat"


def test_triple_quote():
    s = 'my_function(a="""dog""")'
    tool = lark_parse_tool(s)
    assert tool.name == "my_function"
    assert tool.args["a"] == "dog"
