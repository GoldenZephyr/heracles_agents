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

import logging

from sldp.sldp_lang import parse_sldp, sldp_equals


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


with open("dsg_feedforward_prompt.yaml", "r") as fo:
    prompt_yaml = yaml.safe_load(fo)

logger.debug(f"Loaded prompt yaml: {prompt_yaml}")

with open("dsg_experiment.yaml", "r") as fo:
    yml = yaml.safe_load(fo)

exp = ExperimentDefinition(**yml)
logger.debug(f"Loaded experiment: {exp}")


analyzed_questions = []
for question in exp.questions:
    cxt = AgentContext(exp.llm_agent)
    # TODO: prompt text is a function of the experiment config(?)
    prompt_obj = Prompt.from_dict(prompt_yaml)
    prompt_obj.novel_instruction = question.question
    cxt.initialize_agent(prompt_obj)
    success, answer = cxt.run()
    logger.info(f"\nLLM Intermediate Answer: {answer}\n")

    # In this case, there is only one agent sequence. But in the cypher-then-refine
    # case, there are two sequences
    cypher_generation_sequence = AgentSequence(
        description="cypher-producing-agent", responses=cxt.get_agent_responses()
    )
    # TODO: run query

    # TODO: could be different llm_agent?
    cxt2 = AgentContext(exp.llm_agent)
    # TODO: load refinement prompt
    cxt2.initialize_agent(X)
    success, answer = cxt.run()
    logger.info(f"LLM Final Answer: {answer}")

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

    refinement_sequence = AgentSequence(
        description="refinement-agent", responses=cxt2.get_agent_responses()
    )

    sequences = [cypher_generation_sequence, refinement_sequence]

    analysis = QuestionAnalysis(correct=correct, valid_answer_format=valid_sldp)
    aq = AnalyzedQuestion(question=question, sequences=sequences, analysis=analysis)
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


with open("dsgdb_feedforward_out.yaml", "w") as fo:
    fo.write(yaml.dump(aqs.model_dump()))
