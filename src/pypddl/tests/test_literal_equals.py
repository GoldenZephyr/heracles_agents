from pypddl.pddl_goal_types import literal_equals
from pypddl.pddl_goal_parser import lark_parse_pddl_goal


def assert_equal(a, b):
    assert literal_equals(lark_parse_pddl_goal(a), lark_parse_pddl_goal(b))


def assert_unequal(a, b):
    assert not literal_equals(lark_parse_pddl_goal(a), lark_parse_pddl_goal(b))


def test_symbol_equal():
    assert_equal("?a", "?a")


def test_symbol_unequal():
    assert_unequal("?a", "?b")


def test_fact_equal():
    assert_equal("(visited-place a)", "(visited-place a)")


def test_fact_unequal():
    assert_unequal("(visited-place a)", "(visited-place b)")
    assert_unequal("(visited-region a)", "(visited-place a)")
