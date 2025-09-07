import logging
import os
import shutil

import yaml

from heracles_evaluation.experiment_definition import ExperimentDescription
from heracles_evaluation.llm_interface import AnalyzedExperiment
from heracles_evaluation.summarize_results import display_experiment_results

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, force=True)


def load_results(fn):
    with open(fn, "r") as fo:
        yml = yaml.safe_load(fo)

    ae = AnalyzedExperiment(**yml)
    return ae


def load_experiment(fn):
    with open(fn, "r") as fo:
        yml = yaml.safe_load(fo)

    experiment = ExperimentDescription(**yml)
    return experiment


table_to_run = "table1"

tables_base = "tables"
table_dir = f"{tables_base}/{table_to_run}/configurations"
output_dir = f"{tables_base}/{table_to_run}/output"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

table_element_fns = os.listdir(table_dir)

for fn in table_element_fns:
    print("Processing ", fn)
    element_path = os.path.join(table_dir, fn)
    experiment = load_experiment(element_path)
    logger.debug(f"Loaded experiment: {experiment}")

    conf_name = experiment.metadata["config_name"]
    output_filepath = f"{output_dir}/{conf_name}_results.yaml"

    if experiment.metadata.get("skip", False):
        print("  Skipping ", fn)
        ae = AnalyzedExperiment(
            experiment_configurations={}, metadata=experiment.metadata
        )
    elif "prerun_reference" in experiment.metadata:
        parent_table = experiment.metadata["prerun_reference"]["table"]
        parent_fn = experiment.metadata["prerun_reference"]["results_fn"]
        parent_filepath = f"{tables_base}/{parent_table}/output/{parent_fn}"
        logger.info(f"Copying {parent_filepath} --> {output_filepath}")
        shutil.copyfile(parent_filepath, output_filepath)
        continue
    else:
        print("  Processing ", fn)
        results = {}
        for configuration_name, experiment_config in experiment.configurations.items():
            logger.info(f"Testing configuration {configuration_name}")
            analyzed_questions = experiment_config.pipeline.function(experiment_config)

            display_experiment_results(analyzed_questions)
            results[configuration_name] = analyzed_questions

        ae = AnalyzedExperiment(experiment_configurations=results)

    ae.metadata["config_name"] = conf_name
    with open(output_filepath, "w") as fo:
        fo.write(yaml.dump(ae.model_dump()))
