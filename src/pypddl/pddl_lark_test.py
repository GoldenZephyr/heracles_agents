from importlib.resources import as_file, files

from lark import Lark, Transformer

import pypddl


class PddlDomainTransformer(Transformer):
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


def get_pddl_domain_lark_grammar():
    with as_file(files(pypddl).joinpath("pddl_domain.lark")) as path:
        with open(str(path), "r") as fo:
            pddl_domain_grammar = fo.read()
    return pddl_domain_grammar


def lark_parse_pddl_domain(string):
    pddl_domain_grammar = get_pddl_domain_lark_grammar()

    pddl_parser = Lark(
        pddl_domain_grammar,
    )

    T = PddlDomainTransformer()

    tree = pddl_parser.parse(string)
    return T.transform(tree)


if __name__ == "__main__":
    with open("example_domain.pddl", "r") as fo:
        domain = fo.read()
    a = lark_parse_pddl_domain(domain)
