import copy
import logging
import os

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
from heracles_evaluation.pipelines.codegen_utils import (
    execute_generated_code,
    load_dsg,
)
from heracles_evaluation.pipelines.comparisons import evaluate_answer
from heracles_evaluation.pipelines.prompt_utils import get_answer_formatting_guidance

logger = logging.getLogger(__name__)


def generate_prompt(
    question: EvalQuestion,
    agent_config: LlmAgent,
    task_state_context: dict[str] = {},
    api_prompt: str = None,
):
    prompt = copy.deepcopy(agent_config.agent_info.prompt_settings.base_prompt)

    if api_prompt:
        prompt.set_api_prompt(api_prompt)

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

    return prompt


# TODO update this function here
def feedforward_codegen(exp):
    analyzed_questions = []
    # Note this won't work for inserting into the scene graph. To do that a copy.deepcopy will be needed in the loop (not including for efficiency, since loading the scene graph is slow)
    # TODO modify experiment config to include this)
    dsg_filepath = os.path.expandvars(exp.dsg_interface.dsg_filepath)
    dsg_labels_filepath = (
        os.path.expandvars(exp.dsg_interface.dsg_labels_filepath)
        if hasattr(exp.dsg_interface, "dsg_labels_filepath")
        and exp.dsg_interface.dsg_labels_filepath
        else None
    )
    scene_graph = load_dsg(dsg_filepath, dsg_labels_filepath)
    # Set api in prompt
    api_string = exp.dsg_interface.get_dsg_api_prompt()
    for question in exp.questions:
        logger.info(f"\n=======================\nQuestion: {question.question}\n")
        cxt = AgentContext(exp.phases["generate-code"])

        prompt = generate_prompt(
            question, exp.phases["generate-code"], api_prompt=api_string
        )

        cxt.initialize_agent(prompt)
        success, answer = cxt.run()
        logger.info(f"\nLLM Intermediate Answer: {answer}\n")

        codgen_sequence = AgentSequence(
            description="codegen-agent", responses=cxt.get_agent_responses()
        )

        # TODO udpate this
        success, code_results = execute_generated_code(answer, scene_graph)

        cxt2 = AgentContext(exp.phases["refine"])
        refinement_prompt = generate_prompt(
            question,
            exp.phases["refine"],
            {"python_results": code_results, "python_code": answer},
        )

        cxt2.initialize_agent(refinement_prompt)
        success, answer = cxt2.run()
        logger.info(f"LLM Final Answer: {answer}")

        valid_format, correct = evaluate_answer(
            question.correctness_comparator, answer, question.solution
        )

        logger.info(f"\n\nCorrect? {correct}\n\n")

        refinement_sequence = AgentSequence(
            description="refinement-agent", responses=cxt2.get_agent_responses()
        )

        sequences = [codgen_sequence, refinement_sequence]

        n_input_tokens = cxt.initial_input_tokens + cxt2.initial_input_tokens
        n_output_tokens = cxt.total_output_tokens + cxt2.total_output_tokens

        analysis = QuestionAnalysis(
            correct=correct,
            valid_answer_format=valid_format,
            input_tokens=n_input_tokens,
            output_tokens=n_output_tokens,
            n_tool_calls=cxt.n_tool_calls + cxt2.n_tool_calls,
        )
        aq = AnalyzedQuestion(
            question=question, answer=answer, sequences=sequences, analysis=analysis
        )
        analyzed_questions.append(aq)

    aqs = AnalyzedQuestions(analyzed_questions=analyzed_questions)
    return aqs


codegen_phase = PipelinePhase(
    name="generate-code", description="Map question to Python code"
)
refine_phase = PipelinePhase(
    name="refine", description="Map result of executed code to final answer"
)
d = PipelineDescription(
    name="feedforward_codegen",
    description="Single codegen query, then refinement",
    phases=[codegen_phase, refine_phase],
    function=feedforward_codegen,
)

register_pipeline(d)
