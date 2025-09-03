from __future__ import annotations
from dataclasses import dataclass


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
class Fact:
    head: str
    params: list[str]

    def __str__(self):
        return f"({self.head} " + " ".join(self.params) + ")"


@dataclass
class NegatedClause:
    clause: Clause

    def __str__(self):
        return "(not " + str(self.clause) + ")"


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


Clause = NegatedClause | Disjunction | Conjunction
Atomic = Fact | Symbol | NegatedAtomic
