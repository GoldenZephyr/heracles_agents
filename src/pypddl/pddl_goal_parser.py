from importlib.resources import as_file, files

from lark import Lark, Transformer

import pypddl
from pypddl.pddl_goal_types import (
    Bool,
    Conjunction,
    Disjunction,
    Fact,
    NegatedAtomic,
    NegatedClause,
    Symbol,
)


def get_pddl_goal_lark_grammar():
    with as_file(files(pypddl).joinpath("pddl_goal.lark")) as path:
        with open(str(path), "r") as fo:
            pddl_goal_grammar = fo.read()
    return pddl_goal_grammar


class PddlGoalTransformer(Transformer):
    def symbol(self, items):
        return Symbol(str(items[0]))

    def atomic(self, items):
        return items[0]

    def negated_atomic(self, items):
        if isinstance(items[1], Bool):
            return Bool(not items[1].value)
        return NegatedAtomic(items[1])

    def negation(self, items):
        return NegatedClause(items[1])

    def element(self, items):
        return items[0]

    def conjunction(self, items):
        return Conjunction(items[1:])

    def disjunction(self, items):
        return Disjunction(items[1:])

    def clause(self, items):
        return items[0]

    def fact(self, items):
        return Fact(head=str(items[0]), params=[str(i) for i in items[1:]])

    def bool_true(self, items):
        return Bool(True)

    def bool_false(self, items):
        return Bool(False)


pddl_goal_grammar = get_pddl_goal_lark_grammar()
pddl_goal_parser = Lark(pddl_goal_grammar, parser="lalr")
# T = PddlGoalTransformer()


def lark_parse_pddl_goal(string):
    T = PddlGoalTransformer()
    tree = pddl_goal_parser.parse(string)
    return T.transform(tree)
