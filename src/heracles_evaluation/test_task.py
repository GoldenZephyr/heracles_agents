from heracles.prompt_schema import Prompt
import yaml

from heracles_evaluation.experiment_definition import ExperimentDefinition
from heracles_evaluation.llm_interface import AgentContext

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

for question in exp.questions:
    prompt_obj = Prompt.from_dict(prompt_yaml)
    prompt_obj.novel_instruction = question.question
    cxt.initialize_agent(prompt_obj)
    success, answer = cxt.run()
    logger.info(f"\nLLM Answer: {answer}\n")
    try:
        valid_sldp = parse_sldp(answer)
    except Exception:
        logger.warning("Invalid SLDP")
        continue
    correct = sldp_equals(question.solution, answer)
    logger.info(f"\n\nCorrect? {correct}\n\n")
