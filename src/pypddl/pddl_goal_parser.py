from importlib.resources import as_file, files
import pypddl
from lark import Lark, Transformer

from pypddl.pddl_goal_types import Symbol, NegatedAtomic, Conjunction, Disjunction, Fact


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
        return NegatedAtomic(items[0])

    def element(self, items):
        return items[0]

    def conjunction(self, items):
        return Conjunction(items)

    def disjunction(self, items):
        return Disjunction(items)

    def clause(self, items):
        return items[0]

    def fact(self, items):
        return Fact(head=str(items[0]), params=[str(i) for i in items[1:]])


def lark_parse_pddl_goal(string):
    pddl_goal_grammar = get_pddl_goal_lark_grammar()

    pddl_goal_parser = Lark(
        pddl_goal_grammar,
    )
    T = PddlGoalTransformer()
    tree = pddl_goal_parser.parse(string)
    return T.transform(tree)
