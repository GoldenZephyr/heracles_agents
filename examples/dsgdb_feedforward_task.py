# from heracles.prompt_schema import Prompt
import copy

import yaml


from heracles_evaluation.prompt import get_sldp_format_description
from heracles_evaluation.experiment_definition import ExperimentDefinition
from heracles_evaluation.llm_interface import (
    AgentContext,
    AnalyzedQuestion,
    AnalyzedQuestions,
    QuestionAnalysis,
    AgentSequence,
    EvalQuestion,
    LlmAgent,
)

from heracles_evaluation.summarize_results import (
    display_analyzed_question_table,
    summarize_results,
    display_table,
)

import logging

from sldp.sldp_lang import parse_sldp, sldp_equals, get_sldp_type

from heracles.query_interface import Neo4jWrapper

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# TODO: probably make this dynamic dispatched?
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


def query_db(dsgdb_conf, cypher_string):
    with Neo4jWrapper(
        dsgdb_conf.uri,
        (
            dsgdb_conf.username.get_secret_value(),
            dsgdb_conf.password.get_secret_value(),
        ),
        atomic_queries=True,
        print_profiles=False,
    ) as db:
        try:
            query_result = str(db.query(cypher_string))
            return True, query_result
        except Exception as ex:
            print(ex)
            query_result = str(ex)
            return False, query_result


with open("dsg_feedforward_experiment.yaml", "r") as fo:
    yml = yaml.safe_load(fo)

exp = ExperimentDefinition(**yml)
logger.debug(f"Loaded experiment: {exp}")


# TODO: wrap this in a function.
# The interface to the function should declare which
# "phase" tags are necessary to run the function
# The defined experiment will specify this function and the
# tagged phases, and then at "validation time" we can
# check if the specified experiment has the correct phases defined.
analyzed_questions = []
for question in exp.questions:
    cxt = AgentContext(exp.phases["generate-cypher"])

    prompt = generate_prompt(question, exp.phases["generate-cypher"])

    cxt.initialize_agent(prompt)
    success, answer = cxt.run()
    logger.info(f"\nLLM Intermediate Answer: {answer}\n")

    cypher_generation_sequence = AgentSequence(
        description="cypher-producing-agent", responses=cxt.get_agent_responses()
    )

    query_result = query_db(exp.dsg_interface, answer)

    cxt2 = AgentContext(exp.phases["refine"])
    refinement_prompt = generate_prompt(
        question,
        exp.phases["refine"],
        {"cypher_results": query_result, "cypher_query": answer},
    )

    cxt2.initialize_agent(refinement_prompt)
    success, answer = cxt2.run()
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


# TODO: lump the result printing stuff into a function
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
