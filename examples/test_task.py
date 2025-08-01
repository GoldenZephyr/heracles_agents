from heracles.prompt_schema import Prompt
import yaml

from heracles_evaluation.experiment_definition import ExperimentDefinition
from heracles_evaluation.llm_interface import (
    AgentContext,
    AnalyzedQuestion,
    AnalyzedQuestions,
    QuestionAnalysis,
    AgentSequence,
)

from heracles_evaluation.summarize_results import (
    display_analyzed_question_table,
    summarize_results,
    display_table,
)

# TODO: I think if the tool gets exported to the __init__.py we can get rid of this
import heracles_evaluation.tools.canary_favog_tool  # NOQA

import logging

from sldp.sldp_lang import parse_sldp, sldp_equals


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


with open("canary_prompt.yaml", "r") as fo:
    prompt_yaml = yaml.safe_load(fo)

logger.debug(f"Loaded prompt yaml: {prompt_yaml}")

with open("canary_experiment.yaml", "r") as fo:
    yml = yaml.safe_load(fo)

exp = ExperimentDefinition(**yml)
logger.debug(f"Loaded experiment: {exp}")


cxt = AgentContext(exp.llm_agent)

analyzed_questions = []
for question in exp.questions:
    prompt_obj = Prompt.from_dict(prompt_yaml)
    prompt_obj.novel_instruction = question.question
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

column_data_map = {
    "Name": "name",
    "Question": "question",
}
display_analyzed_question_table("Test Table", aqs, column_data_map)


summary_column_data_map = {
    "# Questions": "questions",
}
result_dicts = [q.analysis.model_dump(mode="json") for q in aqs.analyzed_questions]
summary_data = [summarize_results(result_dicts)]
display_table("Summary", summary_data, column_data_map=summary_column_data_map)


with open("test_out.yaml", "w") as fo:
    fo.write(yaml.dump(aqs.model_dump()))
