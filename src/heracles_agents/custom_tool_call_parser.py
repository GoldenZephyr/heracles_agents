from dataclasses import dataclass
from importlib.resources import as_file, files

from lark import Lark, Transformer

import heracles_agents


@dataclass
class FunctionCall:
    name: str
    args: dict[str, float | int | str]


class ToolCallTransformer(Transformer):
    """Transforms a raw SLDP Lark parse tree into
    the Lisp-inspired SLDP representation we use
    for equality checking"""

    def int(self, i):
        return int(i[0])

    def float(self, f):
        return float(f[0])

    def string(self, s):
        return str(s[0][1:-1])

    def triple_string(self, s):
        return str(s[0][3:-3])

    def arg_name(self, n):
        return str(n[0])

    def arg_val(self, v):
        return v[0]

    def kwarg(self, kv):
        return {kv[0]: kv[1]}

    def function_name(self, n):
        return str(n[0])

    def tool_call(self, tc):
        name = tc[0]
        args = {}
        for arg in tc[1:]:
            args |= arg
        return FunctionCall(name=name, args=args)


def get_custom_tool_call_lark_grammar():
    with as_file(files(heracles_agents).joinpath("tool_call.lark")) as path:
        with open(str(path), "r") as fo:
            tool_call_grammar = fo.read()
    return tool_call_grammar


def lark_parse_tool(string):
    tool_grammar = get_custom_tool_call_lark_grammar()

    tool_parser = Lark(
        tool_grammar,
        parser="lalr",
    )

    T = ToolCallTransformer()

    tree = tool_parser.parse(string)
    return T.transform(tree)


if __name__ == "__main__":
    a = lark_parse_tool("""my_function(a=1, b=2, c='cat', d=1.2, e="dog")""")
