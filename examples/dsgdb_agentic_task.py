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
from heracles_evaluation.prompt import get_sldp_format_description
from sldp.sldp_lang import get_sldp_type, parse_sldp, sldp_equals

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

    match agent_config.agent_info.prompt_settings.output_type:
        case "SLDP":
            prompt.answer_formatting_guidance = get_sldp_format_description()
            if agent_config.agent_info.prompt_settings.sldp_answer_type_hint:
                sldp_type = get_sldp_type(question.solution)
                prompt.answer_formatting_guidance += (
                    f"\n Your answer should be an SLDP {sldp_type}"
                )
        case None:
            # The "default". Presumably the description of the output format is
            # in the base prompt.
            pass
        case _:
            raise ValueError(
                f"Unknown output type: {agent_config.prompt_settings.output_type}"
            )

    print("prompt: ")
    print(prompt)
    return prompt


def agentic_cypher_qa(exp):
    analyzed_questions = []
    for question in exp.questions:
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

        try:
            parse_sldp(answer)
            valid_sldp = True
        except Exception:
            logger.warning("Invalid SLDP")
            valid_sldp = False

        if valid_sldp:
            correct = sldp_equals(question.solution, answer)
        else:
            correct = False
        logger.info(f"\n\nCorrect? {correct}\n\n")

        analysis = QuestionAnalysis(correct=correct, valid_answer_format=valid_sldp)
        aq = AnalyzedQuestion(
            question=question, sequences=[agent_sequence], analysis=analysis
        )
        analyzed_questions.append(aq)

    aqs = AnalyzedQuestions(analyzed_questions=analyzed_questions)
    return aqs


main_phase = PipelinePhase(
    name="main",
    description="Map question to (sequence of) Cypher queries, and then to final answer.",
)


d = PipelineDescription(
    name="agentic_cypher_qa",
    description="Multi-query Cypher tool calling",
    phases=[main_phase],
    function=agentic_cypher_qa,
)

register_pipeline(d)

if __name__ == "__main__":
    import yaml

    from heracles_evaluation.experiment_definition import ExperimentConfiguration
    from heracles_evaluation.summarize_results import display_experiment_results

    with open("experiments/dsg_agentic_experiment.yaml", "r") as fo:
        yml = yaml.safe_load(fo)
    experiment = ExperimentConfiguration(**yml)
    logger.debug(f"Loaded experiment configuration: {experiment}")

    aqs = agentic_cypher_qa(experiment)
    with open("output/dsgdb_agentic_out.yaml", "w") as fo:
        fo.write(yaml.dump(aqs.model_dump()))

    display_experiment_results(aqs)
