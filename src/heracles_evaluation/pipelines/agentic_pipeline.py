import copy
import logging

from heracles_evaluation.pipelines.comparisons import evaluate_answer
from heracles_evaluation.pipelines.prompt_utils import get_answer_formatting_guidance

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


def agentic_pipeline(exp):
    analyzed_questions = []
    for question in exp.questions:
        logger.info(f"\n=======================\nQuestion: {question.question}\n")
        cxt = AgentContext(exp.phases["main"])

        prompt = generate_prompt(question, exp.phases["main"])
        logger.info(f"\nLLM Prompt: {prompt}\n")

        cxt.initialize_agent(prompt)
        success, answer = cxt.run()
        logger.info(f"\nLLM Answer: {answer}\n")

        agent_sequence = AgentSequence(
            description="cypher-agent", responses=cxt.get_agent_responses()
        )

        valid_format, correct = evaluate_answer(
            question.correctness_comparator, answer, question.solution
        )

        logger.info(f"\n\nCorrect? {correct}\n\n")

        analysis = QuestionAnalysis(correct=correct, valid_answer_format=valid_format)
        aq = AnalyzedQuestion(
            question=question,
            answer=answer,
            sequences=[agent_sequence],
            analysis=analysis,
        )
        analyzed_questions.append(aq)

    aqs = AnalyzedQuestions(analyzed_questions=analyzed_questions)
    return aqs


main_phase = PipelinePhase(
    name="main",
    description="Use tools to reason about question, and then to submit final answer.",
)


d = PipelineDescription(
    name="agentic",
    description="Agentic pipeline for 3D scene graphs",
    phases=[main_phase],
    function=agentic_pipeline,
)

register_pipeline(d)
