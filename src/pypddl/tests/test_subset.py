# import pytest
from pypddl.pddl_goal_parser import lark_parse_pddl_goal
from pypddl.pddl_goal_types import clause_subset


def assert_subset(formula, expected):
    parsed_expected = lark_parse_pddl_goal(expected)
    parsed_formula = lark_parse_pddl_goal(formula)
    assert clause_subset(parsed_formula, parsed_expected), (
        f"Got {str(parsed_formula)}, Expected {str(parsed_expected)}"
    )


def assert_not_subset(formula, expected):
    parsed_expected = lark_parse_pddl_goal(expected)
    parsed_formula = lark_parse_pddl_goal(formula)
    assert not clause_subset(parsed_formula, parsed_expected), (
        f"Got {str(parsed_formula)}, expected to not be subset of {str(parsed_expected)}"
    )


def test_single_variable():
    assert_subset("?a", "?a")


def test_constant_true():
    assert_subset("True", "True")


def test_constant_false():
    assert_subset("False", "False")


def test_simple_and():
    assert_subset("(and ?a ?b)", "(and ?a ?b)")


def test_simple_or():
    assert_subset("(or ?a ?b)", "(or ?a ?b)")
    assert_not_subset("(or ?a ?b ?c)", "(or ?a ?b)")


def test_dnf_or():
    assert_subset(
        "(or (and ?a ?b) (and ?a ?c))", "(or (and ?d ?e) (and ?a ?b) (and ?a ?c))"
    )

    assert_subset("(and ?a ?b)", "(or (and ?d ?e) (and ?a ?b) (and ?a ?c))")

    assert_not_subset(
        "(or (and ?a ?b ?e) (and ?a ?c))", "(or (and ?d ?e) (and ?a ?b) (and ?a ?c))"
    )


def test_simple_strict_subset_or():
    assert_subset("(or ?a ?b)", "(or ?a ?b ?c)")
