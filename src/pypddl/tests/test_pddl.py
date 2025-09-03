from pypddl.pddl_goal_parser import lark_parse_pddl_goal

from pypddl.pddl_goal_manipulations import (
    pddl_goal_equals,
)


def assert_equal(formula, expected):
    parsed_expected = lark_parse_pddl_goal(expected)
    parsed_formula = lark_parse_pddl_goal(formula)
    assert pddl_goal_equals(
        parsed_formula, parsed_expected
    ), f"Got {str(parsed_formula)}, Expected {str(parsed_expected)}"


def assert_unequal(formula, expected):
    parsed_expected = lark_parse_pddl_goal(expected)
    parsed_formula = lark_parse_pddl_goal(formula)
    assert not pddl_goal_equals(
        parsed_formula, parsed_expected
    ), f"Got {str(parsed_formula)}, expected to not equal {str(parsed_expected)}"


def test_simple():
    assert_equal(
        "(and (visited-place p1) (visited-place p2))",
        "(and (visited-place p2) (visited-place p1))",
    )

    assert_equal(
        "(or (visited-place p1) (visited-place p2))",
        "(or (visited-place p2) (visited-place p1))",
    )

    assert_unequal(
        "(visited-place p1)",
        "(visited-place p2)",
    )

    assert_unequal(
        "(visited-place p1)",
        "(not (visited-place p1))",
    )


def test_simplification():
    assert_equal(
        "(not (not (visited-place p1)))",
        "(visited-place p1)",
    )


def test_demorgan():
    assert_equal("(not (and (vp p1) (vp p2)))", "(or (not (vp p1)) (not (vp p2)))")
