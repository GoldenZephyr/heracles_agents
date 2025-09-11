import logging

import yaml

from heracles_evaluation.experiment_definition import (
    ExperimentDescription,
    PipelineDescription,
    PipelinePhase,
    register_pipeline,
)
from heracles_evaluation.llm_interface import (
    AgentContext,
    AgentSequence,
    AnalyzedQuestion,
    AnalyzedQuestions,
    QuestionAnalysis,
)
from heracles_evaluation.prompt import (
    get_sldp_answer_tag_text,
    get_sldp_format_description,
)
from heracles_evaluation.summarize_results import display_experiment_results
from sldp.sldp_lang import parse_sldp, sldp_equals

logger = logging.getLogger(__name__)


def canary_pipeline(exp):
    cxt = AgentContext(exp.phases["main"])

    analyzed_questions = []
    for question in exp.questions:
        prompt_obj = exp.phases["main"].agent_info.prompt_settings.base_prompt
        prompt_obj.novel_instruction = question.question
        formatting = get_sldp_format_description()
        if exp.phases["main"].agent_info.prompt_settings.output_type != "SLDP_TOOL":
            formatting += get_sldp_answer_tag_text()
        prompt_obj.answer_formatting_guidance = formatting
        cxt.initialize_agent(prompt_obj)
        success, answer = cxt.run()
        logger.info(f"\nLLM Answer: {answer}\n")

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

        # In this case, there is only one agent sequence. But in the cypher-then-refine
        # case, there are two sequences
        agent_sequence = AgentSequence(
            description="tool-calling-agent", responses=cxt.get_agent_responses()
        )
        analysis = QuestionAnalysis(
            correct=correct,
            valid_answer_format=valid_sldp,
            input_tokens=cxt.total_input_tokens,
            output_tokens=cxt.total_output_tokens,
        )

        aq = AnalyzedQuestion(
            question=question,
            sequences=[agent_sequence],
            analysis=analysis,
            answer=answer,
        )
        analyzed_questions.append(aq)

    aqs = AnalyzedQuestions(analyzed_questions=analyzed_questions)
    return aqs


main_phase = PipelinePhase(name="main", description="main canary phase")
d = PipelineDescription(
    name="canary",
    description="For initial testing",
    phases=[main_phase],
    function=canary_pipeline,
)

register_pipeline(d)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with open("experiments/canary_experiment.yaml", "r") as fo:
        yml = yaml.safe_load(fo)

    exp = ExperimentDescription(**yml)
    logger.debug(f"Loaded experiment: {exp}")

    aqs = canary_pipeline(exp)

    with open("output/test_out.yaml", "w") as fo:
        fo.write(yaml.dump(aqs.model_dump()))

    display_experiment_results(aqs)
