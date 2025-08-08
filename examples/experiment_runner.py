from heracles_evaluation.experiment_definition import ExperimentDescription
from heracles_evaluation.summarize_results import display_experiment_results
from heracles_evaluation.llm_interface import AnalyzedExperiment
import heracles_evaluation.tools.canary_favog_tool  # NOQA
import yaml
import logging

import dsgdb_feedforward_task  # NOQA
import test_task  # NOQA

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

with open("master_experiment_anchors.yaml", "r") as fo:
    yml = yaml.safe_load(fo)

experiment = ExperimentDescription(**yml)
logger.debug(f"Loaded experiment: {experiment}")

results = {}
for configuration_name, experiment_config in experiment.configurations.items():
    logger.info(f"Testing configuration {configuration_name}")
    experiment_config.pipeline.validate_agent_phases(
        experiment_config
    )  # TODO: eventually this check should be handled automatically in the construction of the ExperimentConfiguration
    analyzed_questions = experiment_config.pipeline.function(experiment_config)

    display_experiment_results(analyzed_questions)
    results[configuration_name] = analyzed_questions

ae = AnalyzedExperiment(experiment_configurations=results)

with open("master_experiment_out.yaml", "w") as fo:
    fo.write(yaml.dump(ae.model_dump()))
