import yaml

from heracles_evaluation.prompt import get_sldp_format_description
from heracles_evaluation.experiment_definition import (
    ExperimentDescription,
    PipelinePhase,
    PipelineDescription,
    register_pipeline,
)
from heracles_evaluation.llm_interface import (
    AgentContext,
    AnalyzedQuestion,
    AnalyzedQuestions,
    QuestionAnalysis,
    AgentSequence,
)

from heracles_evaluation.summarize_results import display_experiment_results

# TODO: I think if the tool gets exported to the __init__.py we can get rid of this
import heracles_evaluation.tools.canary_favog_tool  # NOQA

import logging

from sldp.sldp_lang import parse_sldp, sldp_equals


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def canary_pipeline(exp):
    cxt = AgentContext(exp.phases["main"])

    analyzed_questions = []
    for question in exp.questions:
        prompt_obj = exp.phases["main"].agent_info.prompt_settings.base_prompt
        prompt_obj.novel_instruction = question.question
        prompt_obj.answer_formatting_guidance = get_sldp_format_description()
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
        analysis = QuestionAnalysis(correct=correct, valid_answer_format=valid_sldp)
        aq = AnalyzedQuestion(
            question=question, sequences=[agent_sequence], analysis=analysis
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
    with open("canary_experiment.yaml", "r") as fo:
        yml = yaml.safe_load(fo)

    exp = ExperimentDescription(**yml)
    logger.debug(f"Loaded experiment: {exp}")

    aqs = canary_pipeline(exp)

    with open("test_out.yaml", "w") as fo:
        fo.write(yaml.dump(aqs.model_dump()))

    display_experiment_results(aqs)
