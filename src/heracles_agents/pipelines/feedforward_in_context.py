import copy
import logging

from heracles_agents.experiment_definition import (
    PipelineDescription,
    PipelinePhase,
    register_pipeline,
)
from heracles_agents.llm_interface import (
    AgentContext,
    AgentSequence,
    AnalyzedQuestion,
    AnalyzedQuestions,
    EvalQuestion,
    LlmAgent,
    QuestionAnalysis,
)
from heracles_agents.pipelines.comparisons import evaluate_answer
from heracles_agents.pipelines.in_context_utils import scene_graph_to_prompt
from heracles_agents.pipelines.prompt_utils import get_answer_formatting_guidance

logger = logging.getLogger(__name__)


def generate_prompt(
    incontext_dsg_interface,
    question: EvalQuestion,
    agent_config: LlmAgent,
    task_state_context: dict[str] = {},
):
    prompt = copy.deepcopy(agent_config.agent_info.prompt_settings.base_prompt)

    dsg_desciption = scene_graph_to_prompt(incontext_dsg_interface.get_dsg())
    try:
        prompt.novel_instruction = prompt.novel_instruction_template.format(
            question=question.question, dsg_description=dsg_desciption
        )
    except KeyError as ex:
        logger.error("Novel instruction template has unfilled parameter!")
        print(ex)
        raise ex

    prompt.answer_semantic_guidance = "Make your answer as concise as possible"
    prompt.answer_formatting_guidance = get_answer_formatting_guidance(
        agent_config, question
    )

    print("prompt: ")
    print(prompt)
    return prompt


def incontext_dsg(exp):
    analyzed_questions = []
    for question in exp.questions:
        try:
            cxt = AgentContext(exp.phases["main"])

            prompt = generate_prompt(exp.dsg_interface, question, exp.phases["main"])

            cxt.initialize_agent(prompt)
            success, answer = cxt.run()
            logger.info(f"\nLLM Final Answer: {answer}\n")

            sequence = AgentSequence(
                description="in-context pipeline", responses=cxt.get_agent_responses()
            )

            valid_format, correct = evaluate_answer(
                question.correctness_comparator, answer, question.solution
            )

            logger.info(f"\n\nCorrect? {correct}\n\n")

            analysis = QuestionAnalysis(
                correct=correct,
                valid_answer_format=valid_format,
                input_tokens=cxt.initial_input_tokens,
                output_tokens=cxt.total_output_tokens,
                n_tool_calls=cxt.n_tool_calls,
            )
        except Exception as ex:
            print(ex)
            logger.error("Bad Question!")
            logger.error(str(ex))
            analysis = QuestionAnalysis(
                correct=False,
                valid_answer_format=False,
                input_tokens=0,
                output_tokens=0,
                n_tool_calls=0,
            )

        aq = AnalyzedQuestion(
            question=question, answer=answer, sequences=[sequence], analysis=analysis
        )
        analyzed_questions.append(aq)

    aqs = AnalyzedQuestions(analyzed_questions=analyzed_questions)
    return aqs


main_phase = PipelinePhase(
    name="main",
    description="Map question to answer using in-context scene graph",
)

d = PipelineDescription(
    name="feedforward_in_context",
    description="in-context scene graph",
    phases=[main_phase],
    function=incontext_dsg,
)

register_pipeline(d)

if __name__ == "__main__":
    import yaml

    from heracles_agents.experiment_definition import ExperimentConfiguration
    from heracles_agents.summarize_results import display_experiment_results

    logging.basicConfig(level=logging.INFO)

    with open("experiments/dsg_incontext_experiment.yaml", "r") as fo:
        yml = yaml.safe_load(fo)
    experiment = ExperimentConfiguration(**yml)
    logger.debug(f"Loaded experiment configuration: {experiment}")

    aqs = incontext_dsg(experiment)
    with open("output/dsgdb_feedforward_out.yaml", "w") as fo:
        fo.write(yaml.dump(aqs.model_dump()))

    display_experiment_results(aqs)
