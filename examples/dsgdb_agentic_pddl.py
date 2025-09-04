import copy
import logging

from prompt_utils import get_answer_formatting_guidance

from heracles_evaluation.experiment_definition import (
    PipelineDescription,
    PipelinePhase,
    register_pipeline,
)
from heracles_evaluation.llm_interface import (
    AgentContext,
    AgentSequence,
    AnalyzedQuestion,
    AnalyzedQuestions,
    EvalQuestion,
    LlmAgent,
    QuestionAnalysis,
)

from pypddl.pddl_goal_manipulations import pddl_goal_equals
from pypddl.pddl_goal_parser import lark_parse_pddl_goal

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def generate_prompt(
    question: EvalQuestion, agent_config: LlmAgent, task_state_context: dict[str] = {}
):
    prompt = copy.deepcopy(agent_config.agent_info.prompt_settings.base_prompt)
    if agent_config.agent_info.tool_interface == "custom":
        prompt.tool_description = "\n".join(
            [t.to_custom() for t in agent_config.agent_info.tools]
        )

    try:
        prompt.novel_instruction = prompt.novel_instruction_template.format(
            question=question.question, **task_state_context
        )
    except KeyError as ex:
        logger.error("Novel instruction template has unfilled parameter!")
        print(ex)
        raise ex

    prompt.answer_semantic_guidance = "Make you answer as concise as possible"
    prompt.answer_formatting_guidance = get_answer_formatting_guidance(
        agent_config, question
    )

    print("prompt: ")
    print(prompt)
    return prompt


def agentic_cypher_pddl(exp):
    analyzed_questions = []
    for question in exp.questions:
        logger.info(f"\n=======================\nQuestion: {question.question}\n")
        cxt = AgentContext(exp.phases["main"])

        prompt = generate_prompt(question, exp.phases["main"])

        cxt.initialize_agent(prompt)
        success, answer = cxt.run()
        logger.info(f"\nLLM Answer: {answer}\n")

        agent_sequence = AgentSequence(
            description="cypher-agent", responses=cxt.get_agent_responses()
        )

        # TODO: In theory, I think the analysis of the correct answer can be
        # handled automatically (i.e., little of the below code here) using the
        # answer comparator that has been defined.

        assert question.correctness_comparator.comparison_type in ["PDDL", "PDDL_TOOL"]

        try:
            parsed_goal = lark_parse_pddl_goal(answer)
            valid_pddl = True
        except Exception as ex:
            print(ex)
            valid_pddl = False

        if valid_pddl:
            correct = pddl_goal_equals(
                parsed_goal, lark_parse_pddl_goal(question.solution)
            )
        else:
            correct = False

        logger.info(f"\n\nCorrect? {correct}\n\n")

        analysis = QuestionAnalysis(correct=correct, valid_answer_format=valid_pddl)
        aq = AnalyzedQuestion(
            question=question, sequences=[agent_sequence], analysis=analysis
        )
        analyzed_questions.append(aq)

    aqs = AnalyzedQuestions(analyzed_questions=analyzed_questions)
    return aqs


main_phase = PipelinePhase(
    name="main",
    description="Map question to (sequence of) Cypher queries, and then to final PDDL answer.",
)


d = PipelineDescription(
    name="agentic_cypher_pddl",
    description="Multi-query Cypher tool calling for PDDL grounding",
    phases=[main_phase],
    function=agentic_cypher_pddl,
)

register_pipeline(d)
