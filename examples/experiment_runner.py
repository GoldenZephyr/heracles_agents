import logging

import dsgdb_agentic_task  # NOQA
import dsgdb_feedforward_task  # NOQA
import incontext_dsg  # NOQA
import test_task  # NOQA
import dsgdb_feedforward_pddl # NOQA
import yaml

import heracles_evaluation.tools.canary_favog_tool  # NOQA
import heracles_evaluation.tools.cypher_query_tool  # NOQA
import heracles_evaluation.tools.sldp_answer_tool  # NOQA
from heracles_evaluation.experiment_definition import ExperimentDescription
from heracles_evaluation.llm_interface import AnalyzedExperiment
from heracles_evaluation.summarize_results import display_experiment_results

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, force=True)

# experiment_fn = "experiments/ollama_test.yaml"
# experiment_fn = "experiments/structured_sldp.yaml"
experiment_fn = "experiments/master_experiment.yaml"
#experiment_fn = "experiments/pddl_experiment.yaml"
with open(experiment_fn, "r") as fo:
    yml = yaml.safe_load(fo)

experiment = ExperimentDescription(**yml)
logger.debug(f"Loaded experiment: {experiment}")

results = {}
for configuration_name, experiment_config in experiment.configurations.items():
    logger.info(f"Testing configuration {configuration_name}")
    analyzed_questions = experiment_config.pipeline.function(experiment_config)

    display_experiment_results(analyzed_questions)
    results[configuration_name] = analyzed_questions

ae = AnalyzedExperiment(experiment_configurations=results)

with open("output/master_experiment_out.yaml", "w") as fo:
    fo.write(yaml.dump(ae.model_dump()))
