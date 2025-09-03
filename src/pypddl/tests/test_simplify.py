from pypddl.pddl_goal_parser import lark_parse_pddl_goal
from pypddl.pddl_goal_types import literal_equals
from pypddl.pddl_goal_manipulations import (
    try_fn,
    simplify,
    flatten_conjunction,
    flatten_disjunction,
    remove_double_negative,
    simplify_singleton_clause,
    simplify_contradiction,
    simplify_tautology,
    evaluate,
)
from functools import partial


def assert_step_equals(fn, formula, expected):
    parsed_expected = lark_parse_pddl_goal(expected)
    parsed_formula = lark_parse_pddl_goal(formula)
    stepped = fn(parsed_formula)
    assert literal_equals(
        stepped, parsed_expected
    ), f"Got {str(stepped)}, Expected {str(parsed_expected)}"


try_simplify = partial(try_fn, simplify)
try_flatten_conjunction = partial(try_fn, flatten_conjunction)
try_flatten_disjunction = partial(try_fn, flatten_disjunction)
try_remove_double_negative = partial(try_fn, remove_double_negative)
try_simplify_singleton = partial(try_fn, simplify_singleton_clause)
try_simplify_contradiction = partial(try_fn, simplify_contradiction)
try_simplify_tautology = partial(try_fn, simplify_tautology)
try_eval = partial(try_fn, evaluate)


def test_single_variable_simplify():
    assert_step_equals(try_simplify, "?a", "?a")
    assert_step_equals(try_simplify, "(visited a)", "(visited a)")


def test_constant_simplify():
    assert_step_equals(try_simplify, "True", "True")
    assert_step_equals(try_simplify, "False", "False")


def test_trivial_clauses():
    assert_step_equals(try_simplify, "(and ?a ?b)", "(and ?a ?b)")
    assert_step_equals(try_simplify, "(or ?a ?b)", "(or ?a ?b)")


def test_flatten():
    assert_step_equals(
        try_flatten_conjunction, "(and (and ?a ?b) (and ?c))", "(and ?a ?b ?c)"
    )
    assert_step_equals(
        try_flatten_conjunction, "(and (and ?a ?b) ?c)", "(and ?a ?b ?c)"
    )
    assert_step_equals(
        try_flatten_disjunction, "(or (or ?a ?b) (or ?c))", "(or ?a ?b ?c)"
    )
    assert_step_equals(try_flatten_disjunction, "(or (or ?a ?b) ?c)", "(or ?a ?b ?c)")


def test_double_negative():
    assert_step_equals(try_remove_double_negative, "(not (not ?a))", "?a")
    assert_step_equals(
        try_remove_double_negative, "(not (not (visited a)))", "(visited a)"
    )
    assert_step_equals(
        try_remove_double_negative, "(not (not (and ?a ?b)))", "(and ?a ?b)"
    )


def test_simplify_singleton():
    assert_step_equals(try_simplify_singleton, "(and ?a)", "?a")
    assert_step_equals(try_simplify_singleton, "(or ?a)", "?a")


def test_contradiction():
    assert_step_equals(try_simplify_contradiction, "(and ?a (not ?a))", "False")
    assert_step_equals(
        try_simplify_contradiction, "(and (visited a) (not (visited a)))", "False"
    )


def test_tautology():
    assert_step_equals(try_simplify_tautology, "(or ?a (not ?a))", "True")
    assert_step_equals(
        try_simplify_tautology, "(or (visited a) (not (visited a)))", "True"
    )


def test_evaluation():
    assert_step_equals(try_eval, "(not True)", "False")
    assert_step_equals(try_eval, "(not False)", "True")

    assert_step_equals(try_eval, "(and True True True)", "True")
    assert_step_equals(try_eval, "(or False False False)", "False")

    assert_step_equals(try_eval, "(or True False)", "True")
    assert_step_equals(try_eval, "(or False False True)", "True")

    assert_step_equals(try_eval, "(and True False)", "False")
    assert_step_equals(try_eval, "(and True False True)", "False")

    assert_step_equals(try_eval, "(and (not True) True)", "False")
    assert_step_equals(try_eval, "(and (not False) True)", "True")

    assert_step_equals(try_eval, "(or (not False) False)", "True")
    assert_step_equals(try_eval, "(or (not True) False)", "False")

    assert_step_equals(try_eval, "(or False False ?a)", "(or ?a)")
    assert_step_equals(try_eval, "(or True ?a)", "True")

    assert_step_equals(try_eval, "(and True ?a)", "(and ?a)")
    assert_step_equals(try_eval, "(and False ?a)", "False")
