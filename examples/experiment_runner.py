import logging

import yaml

from heracles_agents.experiment_definition import ExperimentDescription
from heracles_agents.llm_interface import AnalyzedExperiment
from heracles_agents.summarize_results import display_experiment_results

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, force=True)

# experiment_fn = "experiments/ollama_test.yaml"
# experiment_fn = "experiments/structured_sldp.yaml"
# experiment_fn = "experiments/master_experiment.yaml"
experiment_fn = "experiments/openai_codegen_agentic_test.yaml"
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
