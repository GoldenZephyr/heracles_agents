import copy
import logging

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
from heracles_evaluation.pipelines.comparisons import evaluate_answer
from heracles_evaluation.pipelines.db_utils import query_db
from heracles_evaluation.pipelines.prompt_utils import get_answer_formatting_guidance

logger = logging.getLogger(__name__)


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

    return prompt


def feedforward_cypher(exp):
    analyzed_questions = []
    for question in exp.questions:
        logger.info(f"\n=======================\nQuestion: {question.question}\n")
        cxt = AgentContext(exp.phases["generate-cypher"])

        prompt = generate_prompt(question, exp.phases["generate-cypher"])

        cxt.initialize_agent(prompt)
        success, answer = cxt.run()
        logger.info(f"\nLLM Intermediate Answer: {answer}\n")

        cypher_generation_sequence = AgentSequence(
            description="cypher-producing-agent", responses=cxt.get_agent_responses()
        )

        success, query_result = query_db(exp.dsg_interface, answer)

        cxt2 = AgentContext(exp.phases["refine"])
        refinement_prompt = generate_prompt(
            question,
            exp.phases["refine"],
            {"cypher_results": query_result, "cypher_query": answer},
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

        sequences = [cypher_generation_sequence, refinement_sequence]

        n_input_tokens = cxt.initial_input_tokens + cxt2.initial_input_tokens
        n_output_tokens = cxt.total_output_tokens + cxt2.total_output_tokens

        analysis = QuestionAnalysis(
            correct=correct,
            valid_answer_format=valid_format,
            input_tokens=n_input_tokens,
            output_tokens=n_output_tokens,
            n_tool_calls=cxt.n_tool_calls + cxt2.n_tool_calls,  # Should be 0...
        )
        aq = AnalyzedQuestion(
            question=question, answer=answer, sequences=sequences, analysis=analysis
        )
        analyzed_questions.append(aq)

    aqs = AnalyzedQuestions(analyzed_questions=analyzed_questions)
    return aqs


cypher_phase = PipelinePhase(
    name="generate-cypher", description="Map question to Cypher query"
)
refine_phase = PipelinePhase(
    name="refine", description="Map result of cypher query to final answer"
)
d = PipelineDescription(
    name="feedforward_cypher",
    description="Single cypher query, then refinement",
    phases=[cypher_phase, refine_phase],
    function=feedforward_cypher,
)

register_pipeline(d)

if __name__ == "__main__":
    import yaml

    from heracles_evaluation.experiment_definition import ExperimentConfiguration
    from heracles_evaluation.summarize_results import display_experiment_results

    logging.basicConfig(level=logging.INFO)

    with open("experiments/dsg_feedforward_experiment.yaml", "r") as fo:
        yml = yaml.safe_load(fo)
    experiment = ExperimentConfiguration(**yml)
    logger.debug(f"Loaded experiment configuration: {experiment}")

    aqs = feedforward_cypher(experiment)
    with open("output/dsgdb_feedforward_out.yaml", "w") as fo:
        fo.write(yaml.dump(aqs.model_dump()))

    display_experiment_results(aqs)
