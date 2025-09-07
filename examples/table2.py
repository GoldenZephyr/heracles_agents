import os
from table_utils import get_method_to_pipeline
from tables import get_questions_fn, get_pipeline, get_interface_for_method, get_phases
from heracles_evaluation.experiment_definition import (
    ExperimentConfiguration,
    ExperimentDescription,
)
import yaml

# +---------------------+---------------+-------+-------+-------+-------+-------+-------+
# |                     |               |      Q&A      |      PDDL     |       SG      |
# |       Method        |    Model      |   S   |   L   |   S   |   L   |   S   |   L   |
# +---------------------+---------------+-------+-------+-------+-------+-------+-------+
# | Cypher (A)          | GPT-4.1       |   x   | 0.96  |   x   | 0.92  |   x   |   x   |
# |                     | GPT-4.1-mini  |   x   | 0.76  |   x   |   x   |   x   |   x   |
# |                     | GPT-4.1-nano  |   x   | 0.32  |   x   |   x   |   x   |   x   |
# +---------------------+---------------+-------+-------+-------+-------+-------+-------+
# | Context Window (A)  | GPT-4.1       |   x   |   x   |   x   |   x   |   x   |   x   |
# |                     | GPT-4.1-mini  |   x   |   x   |   x   |   x   |   x   |   x   |
# |                     | GPT-4.1-nano  |   x   |   x   |   x   |   x   |   x   |   x   |
# +---------------------+---------------+-------+-------+-------+-------+-------+-------+
# | Python (A)          | GPT-4.1       |   x   |   x   |   x   |   x   |   x   |   x   |
# |                     | GPT-4.1-mini  |   x   |   x   |   x   |   x   |   x   |   x   |
# |                     | GPT-4.1-nano  |   x   |   x   |   x   |   x   |   x   |   x   |
# +---------------------+---------------+-------+-------+-------+-------+-------+-------+


prompt_base = "prompts"
output_base = "tables/table2/configurations"
if not os.path.exists(output_base):
    print(f"{output_base} does not exist, creating it!")
    os.makedirs(output_base)

dsg_paths = {}
dsg_paths["b45"] = (
    "$HERACLES_EVALUATION_PATH/examples/scene_graphs/2025-09-04-heracles-eval-2_dsg_with_mesh.json"
)
dsg_paths["westpoint"] = (
    "$HERACLES_EVALUATION_PATH/examples/scene_graphs/west_point_fused_map_wregions_labelspace.json"
)

dsg_tags = ["westpoint", "b45"]

methods = [
    "agentic_cypher",
    "in_context",
    "agentic_in_context",
    "agentic_python",
]
tasks = ["qa", "pddl", "update_qa"]


pipelines = [
    "agentic",
    "feedforward_cypher",
    "feedforward_in_context",
    "feedforward_python",
]

# models = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]
models = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]


prerun_configs = {}

model_from_table1 = "gpt-4.1"
# model_from_table1 = "gpt-4.1-nano"
for method in ["agentic_cypher", "in_context", "agentic_in_context", "agentic_python"]:
    for task in ["qa", "pddl", "update_qa"]:
        for tag in ["westpoint", "b45"]:
            prerun_configs[f"{method}_{model_from_table1}_{task}_{tag}"] = {
                "table": "table1",
                "results_fn": f"{method}_{model_from_table1}_{task}_{tag}_results.yaml",
            }


method_to_pipeline = get_method_to_pipeline()


for m in methods:
    for model in models:
        for t in tasks:
            for tag in dsg_tags:
                config_name = f"{m}_{model}_{t}_{tag}"
                print("config_name: ", config_name)
                configurations = {}
                metadata = {"config_name": config_name}
                if config_name in prerun_configs:
                    metadata["prerun_reference"] = prerun_configs[config_name]
                else:
                    questions_fn = get_questions_fn(t, tag)
                    if os.path.exists(os.path.expandvars(questions_fn)):
                        try:
                            config = ExperimentConfiguration(
                                pipeline=get_pipeline(m),
                                phases=get_phases(m, t, model=model),
                                dsg_interface=get_interface_for_method(m, tag),
                                questions=questions_fn,
                            )
                            print("config: ", config)
                            metadata["dsg_tag"] = tag
                            configurations[config_name] = config
                        except NotImplementedError as ex:
                            print(ex)
                            metadata["skip"] = True
                            metadata["going_to_implement"] = True

                        except FileNotFoundError as ex:
                            print(ex)
                            metadata["skip"] = True
                            metadata["going_to_implement"] = True

                    else:
                        print(
                            f"No questions for configuration ({questions_fn}). Skipping"
                        )
                        metadata["skip"] = True
                        metadata["going_to_implement"] = True

                exp = ExperimentDescription(
                    metadata=metadata,
                    configurations=configurations,
                )

                output_fn = f"{output_base}/{config_name}.yaml"
                print(f"Saving to {output_fn}")
                with open(output_fn, "w") as fo:
                    fo.write(yaml.dump(exp.model_dump()))
