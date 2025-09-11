# ruff: noqa: F811
import logging

from plum import dispatch

from heracles_evaluation.llm_interface import PddlComparison, SldpComparison
from pypddl.pddl_goal_manipulations import pddl_goal_equals
from pypddl.pddl_goal_parser import lark_parse_pddl_goal
from sldp.sldp_lang import lark_parse_sldp, sldp_equals

logger = logging.getLogger(__name__)


@dispatch
def evaluate_answer(comparator: PddlComparison, answer, solution):
    try:
        parsed_goal = lark_parse_pddl_goal(answer)
        valid_pddl = True
    except Exception as ex:
        print(ex)
        logger.warning("Invalid PDDL goal")
        valid_pddl = False

    if valid_pddl:
        correct = pddl_goal_equals(parsed_goal, lark_parse_pddl_goal(solution))
    else:
        correct = False

    return valid_pddl, correct


@dispatch
def evaluate_answer(comparator: SldpComparison, answer, solution):
    try:
        lark_parse_sldp(answer)
        valid_sldp = True
    except Exception as ex:
        print(ex)
        logger.warning("Invalid SLDP")
        valid_sldp = False

    if valid_sldp:
        correct = sldp_equals(solution, answer)
    else:
        correct = False
    return valid_sldp, correct
