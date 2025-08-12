from lark import Lark, Transformer


class SldpTransformer(Transformer):
    def float(self, items):
        return float(items[0])

    def string(self, s):
        return str(s[0])

    def point(self, coords):
        return ("point", *coords)

    def list(self, items):
        return ("list", *items)

    def kv_pair(self, kv):
        k, v = kv
        return ("pair", k, v)

    def dict(self, items):
        return ("dict", *items)

    def set(self, items):
        return ("set", *items)


def lark_parse_sldp(string):
    sldp_parser = Lark(
        r"""
    ?expression: float | string | set | list | dict | point
    string: CNAME
    float: FLOAT | INT
    set: "<" [expression ("," expression)*] ">"
    list: "[" [expression ("," expression)*] "]"
    kv_pair: string ":" expression
    dict: "{" [kv_pair] ("," kv_pair)* "}"
    point: "POINT(" float float float ")"

    %import common.FLOAT
    %import common.LETTER
    %import common.INT
    %import common.WS
    %import common.CNAME
    %ignore WS
    """,
        start="expression",
    )

    T = SldpTransformer()

    tree = sldp_parser.parse(string)
    return T.transform(tree)


if __name__ == "__main__":

    a = lark_parse_sldp("<1, 2, 3>")
