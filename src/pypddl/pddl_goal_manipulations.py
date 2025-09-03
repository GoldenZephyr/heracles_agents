# Turn off warnings for duplicated function definitions
# ruff: noqa: F811
from __future__ import annotations
from plum import dispatch

from pypddl.pddl_goal_types import (
    Disjunction,
    Conjunction,
    NegatedClause,
    Clause,
    NegatedAtomic,
    Fact,
)

from pypddl.pddl_goal_parser import lark_parse_pddl_goal


### Negate
@dispatch
def negate(clause: Disjunction | Conjunction | NegatedClause):
    return NegatedClause(clause)


@dispatch
def negate(clause):
    return False


@dispatch
def negate(fact: Fact):
    return NegatedAtomic(fact)


@dispatch
def negate(fact: NegatedAtomic):
    return fact.atomic


### Demorgan
@dispatch
def demorgan(clause):
    return False


def demorgan(clause: Clause):
    # (not (or x y)) -> (and (not x) (not y))
    # (not (and x y)) -> (or (not x) (not y))
    match clause:
        case NegatedClause():
            inner_clause = clause.clause
            match inner_clause:
                case Disjunction():
                    return Conjunction([negate(c) for c in inner_clause.clauses])
                case Conjunction():
                    return Disjunction([negate(c) for c in inner_clause.clauses])
        case _:
            return False


### Distribute Conjunction
@dispatch
def distribute_conjunction(clause):
    # (and x (or y z)) -> (or (and x y) (and x z)) (distribute conjunction)
    # (and (or x y) z) -> (or (and x z) (and y z)) (distribute conjunction)
    pass


### Distribute Disjunction


@dispatch
def remove_double_negative(clause):
    return False


@dispatch
def remove_double_negative(clause: NegatedClause):
    match clause.clause:
        case NegatedClause():
            return clause.clause.clause
        case NegatedAtomic():
            return clause.clause.atomic
        case _:
            return False


@dispatch
def remove_double_negative(clause: NegatedAtomic):
    match clause.atomic:
        case NegatedAtomic():
            return clause.atomic.atomic
        case _:
            return False


@dispatch
def simplify_singleton_clause(clause):
    return False


@dispatch
def simplify_singleton_clause(clause: Conjunction | Disjunction):
    if len(clause.clauses) == 1:
        return clause.clauses[0]
    return False


def try_fn(fn, clause):
    c = fn(clause)
    if c:
        return c
    return clause


def simplify_step(clause):
    clause = try_fn(remove_double_negative, clause)
    clause = try_fn(simplify_singleton_clause, clause)
    return clause


def simplify(clause):
    # TODO: loop until no more changes
    return simplify_step(clause)


def make_dnf_inner(clause):
    clause = simplify(clause)
    clause = try_fn(demorgan, clause)
    clause = try_fn(distribute_conjunction, clause)
    return clause


test = "(or (and ?a ?b) (and ?c ?d) (and (not ?a) (not ?d)))"

a = lark_parse_pddl_goal(test)
print(a)
test2 = "(not (not ?a))"
b = lark_parse_pddl_goal(test2)
print(b)
test3 = "(not (visited-object o1))"
c = lark_parse_pddl_goal(test3)

test4 = "(and (not (not (visited-object o1))))"
d = lark_parse_pddl_goal(test4)
