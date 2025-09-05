import logging

import agentic_pipeline  # NOQA
import feedforward_cypher_pipeline  # NOQA
import feedforward_in_context  # NOQA
import test_task  # NOQA
import yaml

import heracles_evaluation.tools.canary_favog_tool  # NOQA
import heracles_evaluation.tools.cypher_query_tool  # NOQA
import heracles_evaluation.tools.sldp_answer_tool  # NOQA
from heracles_evaluation.experiment_definition import ExperimentDescription
from heracles_evaluation.llm_interface import AnalyzedExperiment
from heracles_evaluation.summarize_results import display_experiment_results
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, force=True)


def load_experiment(fn):
    with open(fn, "r") as fo:
        yml = yaml.safe_load(fo)

    experiment = ExperimentDescription(**yml)
    return experiment


table_dir = "tables/table1/configurations"
output_dir = "tables/table1/output"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

table_element_fns = os.listdir(table_dir)

for fn in table_element_fns:
    if "context" not in fn:
        continue
    print("Processing ", fn)
    element_path = os.path.join(table_dir, fn)
    experiment = load_experiment(element_path)
    logger.debug(f"Loaded experiment: {experiment}")

    if experiment.metadata.get("skip", False):
        print("  Skipping ", fn)
        ae = AnalyzedExperiment(
            experiment_configurations={}, metadata=experiment.metadata
        )
    else:
        print("  Processing ", fn)
        results = {}
        for configuration_name, experiment_config in experiment.configurations.items():
            logger.info(f"Testing configuration {configuration_name}")
            analyzed_questions = experiment_config.pipeline.function(experiment_config)

            display_experiment_results(analyzed_questions)
            results[configuration_name] = analyzed_questions

        ae = AnalyzedExperiment(experiment_configurations=results)

    conf_name = experiment.metadata["config_name"]
    ae.metadata["config_name"] = conf_name
    with open(f"{output_dir}/{conf_name}_results.yaml", "w") as fo:
        fo.write(yaml.dump(ae.model_dump()))
