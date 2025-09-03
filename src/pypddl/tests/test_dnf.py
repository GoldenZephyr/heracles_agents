# import pytest
from pypddl.pddl_goal_manipulations import convert_to_dnf
from pypddl.pddl_goal_parser import lark_parse_pddl_goal
from pypddl.pddl_goal_types import clause_equals


def assert_dnf_equal(formula, expected):
    parsed_expected = lark_parse_pddl_goal(expected)
    parsed_formula = lark_parse_pddl_goal(formula)
    dnf = convert_to_dnf(parsed_formula)
    assert clause_equals(
        dnf, parsed_expected
    ), f"Got {str(dnf)}, Expected {str(parsed_expected)}"


def test_single_variable():
    assert_dnf_equal("?a", "?a")


def test_constant_true():
    assert_dnf_equal("True", "True")


def test_constant_false():
    assert_dnf_equal("False", "False")


def test_simple_and():
    assert_dnf_equal("(and ?a ?b)", "(and ?a ?b)")


def test_simple_or():
    assert_dnf_equal("(or ?a ?b)", "(or ?a ?b)")


def test_negation_pushdown():
    assert_dnf_equal("(not (and ?a ?b))", "(or (not ?a) (not ?b))")


def test_distribution():
    assert_dnf_equal("(and ?a (or ?b ?c))", "(or (and ?a ?b) (and ?a ?c))")


def test_nested_distribution():
    formula = "(or (and ?a ?b) (and ?c ?d) (and (not ?a) (not ?d)))"
    expected = "(or (and ?a ?b) (and ?c ?d) (and (not ?a) (not ?d)))"
    assert_dnf_equal(formula, expected)


def test_double_negation():
    assert_dnf_equal("(not (not ?a))", "?a")


def test_deeply_nested_expression():
    formula = "(and (or ?a (and ?b ?c)) (or ?d ?e))"
    expected = "(or (and ?a ?d) (and ?a ?e) (and ?b ?c ?d) (and ?b ?c ?e))"
    assert_dnf_equal(formula, expected)
