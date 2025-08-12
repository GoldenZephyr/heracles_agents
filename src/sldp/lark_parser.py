from lark import Lark, Transformer
import sldp
from importlib.resources import as_file, files


class SldpTransformer(Transformer):
    """Transforms a raw SLDP Lark parse tree into
    the Lisp-inspired SLDP representation we use
    for equality checking"""

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


def get_sldp_lark_grammar():
    with as_file(files(sldp).joinpath("sldp.lark")) as path:
        with open(str(path), "r") as fo:
            sldp_grammar = fo.read()
    return sldp_grammar


def lark_parse_sldp(string):
    sldp_grammar = get_sldp_lark_grammar()

    sldp_parser = Lark(
        sldp_grammar,
    )

    T = SldpTransformer()

    tree = sldp_parser.parse(string)
    return T.transform(tree)


if __name__ == "__main__":
    a = lark_parse_sldp("<1, 2, 3>")
