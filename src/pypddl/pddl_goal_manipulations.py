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
    Symbol,
    Fact,
    fmap,
    Atomic,
    literal_equals,
    Bool,
    clause_equals,
)

from pypddl.pddl_goal_parser import lark_parse_pddl_goal
from functools import partial


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
def negate(symbol: Symbol):
    return NegatedAtomic(symbol)


@dispatch
def negate(fact: NegatedAtomic):
    return fact.atomic


@dispatch
def negate(b: Bool):
    return Bool(not b.value)


### Demorgan
@dispatch
def demorgan(clause):
    return False


def demorgan(clause: Clause):
    match clause:
        case NegatedClause():
            inner_clause = clause.clause
            match inner_clause:
                case Disjunction():
                    return Conjunction([negate(c) for c in inner_clause.clauses])
                case Conjunction():
                    return Disjunction([negate(c) for c in inner_clause.clauses])
                case _:
                    raise NotImplementedError(
                        f"demorgan can't process type {type(inner_clause)}"
                    )
        case _:
            return False


### Flatten


@dispatch
def flatten_conjunction(x):
    return False


@dispatch
def flatten_disjunction(x):
    return False


@dispatch
def flatten_conjunction(conjunction: Conjunction):
    found_child_conjunction = False
    new_clauses = []
    for c in conjunction.clauses:
        if isinstance(c, Conjunction):
            found_child_conjunction = True
            new_clauses += c.clauses
        else:
            new_clauses.append(c)
    if found_child_conjunction:
        return Conjunction(new_clauses)
    return False


@dispatch
def flatten_disjunction(disjunction: Disjunction):
    found_child_disjunction = False
    new_clauses = []
    for c in disjunction.clauses:
        if isinstance(c, Disjunction):
            found_child_disjunction = True
            new_clauses += c.clauses
        else:
            new_clauses.append(c)
    if found_child_disjunction:
        return Disjunction(new_clauses)
    return False


### Distribute Conjunction
@dispatch
def distribute_conjunction(clause):
    return False


@dispatch
def distribute_conjunction(conjunction: Conjunction):
    ors = [c for c in conjunction.clauses if isinstance(c, Disjunction)]
    if len(ors) == 0:
        return False
    rest = [c for c in conjunction.clauses if not isinstance(c, Disjunction)]

    disjunction_to_pull = ors[0]

    c1 = Conjunction(rest + ors[1:] + [disjunction_to_pull.clauses[0]])
    c2 = Conjunction(rest + ors[1:] + disjunction_to_pull.clauses[1:])

    return Disjunction([c1, c2])


### Distribute Disjunction
@dispatch
def distribute_disjunction(clause):
    return False


@dispatch
def distribute_disjunction(disjunction: Disjunction):
    raise NotImplementedError("TODO")


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
def simplify_contradiction(clause):
    return False


@dispatch
def simplify_contradiction(conjunction: Conjunction):
    atomics = [
        c
        for c in conjunction.clauses
        if any(isinstance(c, t) for t in [Atomic, NegatedAtomic])
    ]
    for a in atomics:
        for b in atomics:
            if literal_equals(a, negate(b)):
                return Bool(False)
    return False


@dispatch
def simplify_tautology(clause):
    return False


@dispatch
def simplify_tautology(disjunction: Disjunction):
    atomics = [
        c
        for c in disjunction.clauses
        if any(isinstance(c, t) for t in [Atomic, NegatedAtomic])
    ]
    for a in atomics:
        for b in atomics:
            if literal_equals(a, negate(b)):
                return Bool(True)
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


@dispatch
def evaluate(clause):
    return False


@dispatch
def evaluate(conjunction: Conjunction):
    bools = [b.value for b in conjunction.clauses if isinstance(b, Bool)]
    rest = [c for c in conjunction.clauses if not isinstance(c, Bool)]
    if len(bools) == 0:
        return False
    if False in bools:
        return Bool(False)
    if len(rest) > 0:
        return Conjunction(rest)
    return Bool(True)


@dispatch
def evaluate(disjunction: Disjunction):
    print("eval")
    bools = [b.value for b in disjunction.clauses if isinstance(b, Bool)]
    print(bools)
    rest = [c for c in disjunction.clauses if not isinstance(c, Bool)]
    print(rest)
    if len(bools) == 0:
        return False
    if True in bools:
        return Bool(True)
    if len(rest) > 0:
        return Disjunction(rest)
    return Bool(False)


@dispatch
def evaluate(na: NegatedAtomic):
    if isinstance(na.atomic, Bool):
        return Bool(not na.atomic.value)
    return False


@dispatch
def simplify_step(clause):
    return False


@dispatch
def simplify_step(clause: Clause | Atomic):
    clause = fmap(partial(try_fn, simplify_step), clause)
    clause = try_fn(remove_double_negative, clause)
    clause = try_fn(simplify_singleton_clause, clause)
    clause = try_fn(flatten_conjunction, clause)
    clause = try_fn(flatten_disjunction, clause)
    clause = try_fn(evaluate, clause)
    return clause


def simplify(clause):
    # TODO: loop until no more changes
    return simplify_step(clause)


def simplify_string(s):
    return simplify(lark_parse_pddl_goal(s))


@dispatch
def make_dnf_inner(clause):
    return False


@dispatch
def make_dnf_inner(clause: Clause | NegatedAtomic):
    clause = fmap(partial(try_fn, make_dnf_inner), clause)
    clause = try_fn(simplify, clause)
    clause = try_fn(demorgan, clause)
    clause = try_fn(distribute_conjunction, clause)
    return clause


def convert_to_dnf(clause):
    new_clause = try_fn(make_dnf_inner, clause)
    if not literal_equals(new_clause, clause):
        return convert_to_dnf(new_clause)
    return clause


@dispatch
def make_cnf_inner(clause):
    return False


@dispatch
def make_cnf_inner(clause: Clause | NegatedAtomic):
    clause = fmap(partial(try_fn, make_dnf_inner), clause)
    clause = try_fn(simplify, clause)
    clause = try_fn(demorgan, clause)
    clause = try_fn(distribute_disjunction, clause)
    return clause


def convert_to_cnf(clause):
    new_clause = try_fn(make_cnf_inner, clause)
    if not literal_equals(new_clause, clause):
        return convert_to_cnf(new_clause)
    return clause


@dispatch
def make_nnf_inner(clause):
    return False


@dispatch
def make_nnf_inner(clause: Clause | NegatedAtomic):
    clause = fmap(partial(try_fn, make_nnf_inner), clause)
    clause = try_fn(simplify, clause)
    clause = try_fn(demorgan, clause)
    return clause


def convert_to_nnf(clause):
    new_clause = try_fn(make_nnf_inner, clause)
    if not literal_equals(new_clause, clause):
        return convert_to_nnf(new_clause)
    return clause


def pddl_goal_equals(a, b):
    _a = convert_to_dnf(a)
    _b = convert_to_dnf(b)
    return clause_equals(simplify(_a), simplify(_b))


# test = "(or (and ?a ?b) (and ?c ?d) (and (not ?a) (not ?d)))"
#
# a = lark_parse_pddl_goal(test)
# print(a)
# test2 = "(not (not ?a))"
# b = lark_parse_pddl_goal(test2)
## print(b)
## test3 = "(not (visited-object o1))"
## c = lark_parse_pddl_goal(test3)
#
## test4 = "(and (not (not (visited-object o1))))"
## d = lark_parse_pddl_goal(test4)
##
## test5 = "(and (or ?a (and ?b ?c)) (or ?d ?e))"
## e = lark_parse_pddl_goal(test5)
##
## convert_to_dnf(e)
#
#
# f = lark_parse_pddl_goal("(or False False False)")
# a = simplify(f)
#
# g1 = lark_parse_pddl_goal("(and (visited-place p1) (visited-place p2))")
# g2 = lark_parse_pddl_goal("(and (visited-place p2) (visited-place p1))")
#
# ans = pddl_goal_equals(g1, g2)
