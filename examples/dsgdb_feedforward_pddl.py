import copy
import logging

from db_utils import query_db
from prompt_utils import get_answer_formatting_guidance

from heracles_evaluation.experiment_definition import (
    PipelineDescription,
    PipelinePhase,
    register_pipeline,
    ExperimentConfiguration,
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

    prompt.answer_semantic_guidance = "Make your answer as concise as possible"
    formatting = get_answer_formatting_guidance(agent_config, question)
    if formatting is not None:
        prompt.answer_formatting_guidance = formatting

    print("prompt: ")
    print(prompt)
    return prompt


def feedforward_cypher_pddl(exp: ExperimentConfiguration):
    analyzed_questions = []
    for question in exp.questions:
        logger.info(f"\n=======================\nQuestion: {question.question}\n")
        cxt = AgentContext(exp.phases["generate-cypher"])

        prompt = generate_prompt(question, exp.phases["generate-cypher"])
        print("LLM prompt: ", prompt)

        cxt.initialize_agent(prompt)
        success, answer = cxt.run()
        logger.info(f"\nLLM Intermediate Answer: {answer}\n")

        cypher_generation_sequence = AgentSequence(
            description="cypher-producing-agent", responses=cxt.get_agent_responses()
        )

        success, query_result = query_db(exp.dsg_interface, answer)

        cxt2 = AgentContext(exp.phases["refine-to-pddl"])
        refinement_prompt = generate_prompt(
            question,
            exp.phases["refine-to-pddl"],
            {"cypher_results": query_result, "cypher_query": answer},
        )
        print("refinement prompt: ", refinement_prompt)

        cxt2.initialize_agent(refinement_prompt)
        success, answer = cxt2.run()
        logger.info(f"LLM Final Answer: {answer}")

        assert question.correctness_comparator.comparison_type in ["PDDL", "PDDL_TOOL"]

        try:
            parsed_goal = lark_parse_pddl_goal(answer)
            valid_pddl = True
        except Exception as ex:
            print(ex)
            valid_pddl = False

        if valid_pddl:
            print("answer  : ", answer)
            print("solution: ", question.solution)
            correct = pddl_goal_equals(
                parsed_goal, lark_parse_pddl_goal(question.solution)
            )
            print("correct: ", correct)
        else:
            correct = False

        refinement_sequence = AgentSequence(
            description="refinement-agent", responses=cxt2.get_agent_responses()
        )

        sequences = [cypher_generation_sequence, refinement_sequence]

        analysis = QuestionAnalysis(correct=correct, valid_answer_format=valid_pddl)
        aq = AnalyzedQuestion(question=question, sequences=sequences, analysis=analysis)
        analyzed_questions.append(aq)

    aqs = AnalyzedQuestions(analyzed_questions=analyzed_questions)
    return aqs


cypher_phase = PipelinePhase(
    name="generate-cypher", description="Map question to Cypher query"
)
refine_phase = PipelinePhase(
    name="refine-to-pddl", description="Map result of cypher query to final answer"
)
d = PipelineDescription(
    name="feedforward_cypher_pddl",
    description="Single cypher query, then refinement to PDDL",
    phases=[cypher_phase, refine_phase],
    function=feedforward_cypher_pddl,
)

register_pipeline(d)
