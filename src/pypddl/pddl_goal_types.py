# ruff: noqa: F811
from __future__ import annotations
from dataclasses import dataclass
from plum import dispatch
from beartype.typing import Callable, List, Set, Iterable


@dataclass
class Disjunction:
    clauses: list[Clause | Atomic]

    def __str__(self):
        return "(or " + " ".join([str(e) for e in self.clauses]) + ")"


@dataclass
class Conjunction:
    clauses: list[Clause | Atomic]

    def __str__(self):
        return "(and " + " ".join([str(e) for e in self.clauses]) + ")"


@dataclass
class NegatedClause:
    clause: Clause

    def __str__(self):
        return "(not " + str(self.clause) + ")"


@dataclass
class Fact:
    head: str
    params: list[str]

    def __str__(self):
        return f"({self.head} " + " ".join(self.params) + ")"


@dataclass
class NegatedAtomic:
    atomic: Atomic

    def __str__(self):
        return "(not " + str(self.atomic) + ")"


@dataclass
class Symbol:
    name: str

    def __str__(self):
        return "?" + self.name


@dataclass
class Bool:
    value: bool

    def __str__(self):
        return str(self.value)


Clause = NegatedClause | Disjunction | Conjunction
Atomic = Bool | Fact | Symbol | NegatedAtomic


@dispatch
def literal_equals(a, b):
    return False


@dispatch
def literal_equals(a: list, b: list):
    return all(literal_equals(_a, _b) for _a, _b in zip(a, b))


@dispatch
def literal_equals(a: str, b: str):
    return a == b


@dispatch
def literal_equals(a: Bool, b: Bool):
    return a.value == b.value


@dispatch
def literal_equals(a: Disjunction | Conjunction, b: Disjunction | Conjunction):
    return type(a) is type(b) and literal_equals(a.clauses, b.clauses)


@dispatch
def literal_equals(a: NegatedClause, b: NegatedClause):
    return literal_equals(a.clause, b.clause)


@dispatch
def literal_equals(a: Fact, b: Fact):
    return a.head == b.head and literal_equals(a.params, b.params)


@dispatch
def literal_equals(a: NegatedAtomic, b: NegatedAtomic):
    return literal_equals(a.atomic, b.atomic)


@dispatch
def literal_equals(a: Symbol, b: Symbol):
    return literal_equals(a.name, b.name)


# @dispatch
# def dnf_equals(a, b):
#    literal_equals(a, b)
#
#
# @dispatch
# def dnf_equals(a: Disjunction, b: Disjunction):
#    # every clause in a is in b. every clause in b is in a
#    for ca in a.clauses:
#        any(clause_equals(ca, cb) for cb in b.clauses)


# clause_equals checks for syntactic equality, but allows for reordering within conjunctions / disjunctions
@dispatch
def clause_equals(a, b):
    return literal_equals(a, b)


@dispatch
def clause_equals(a: Disjunction | Conjunction, b: Disjunction | Conjunction):
    if type(a) is not type(b):
        return False
    # Every a is in b
    for ca in a.clauses:
        if not any(clause_equals(ca, cb) for cb in b.clauses):
            print(f"{ca} not in {b.clauses}")
            return False
    # Every b is in a
    for cb in b.clauses:
        if not any(clause_equals(ca, cb) for ca in a.clauses):
            print(f"{cb} not in {a.clauses}")
            return False
    return True


@dispatch
def clause_equals(a: NegatedClause, b: NegatedClause):
    return clause_equals(a.clause, b.clause)


@dispatch
def fmap(fn: Callable, iterable: Iterable):
    raise Exception(
        f"fmap not implemented for {type(iterable)}, but you can implement it!"
    )


@dispatch
def fmap(fn: Callable, lst: List):
    return [fn(e) for e in lst]


@dispatch
def fmap(fn: Callable, s: Set):
    return set(fn(e) for e in s)


@dispatch
def fmap(fn: Callable, clause: Disjunction):
    return Disjunction(list(map(fn, clause.clauses)))


@dispatch
def fmap(fn: Callable, clause: Conjunction):
    return Conjunction(list(map(fn, clause.clauses)))


@dispatch
def fmap(fn: Callable, clause: NegatedClause):
    return NegatedClause(fn(clause.clause))


@dispatch
def fmap(fn: Callable, clause: Fact):
    return Fact(fn(clause.head), list(map(fn, clause.params)))


@dispatch
def fmap(fn: Callable, clause: NegatedAtomic):
    return NegatedAtomic(fmap(fn, clause.atomic))


@dispatch
def fmap(fn: Callable, clause: Symbol):
    return Symbol(fn(clause.name))


@dispatch
def fmap(fn: Callable, clause: Bool):
    return Bool(fn(clause.value))
