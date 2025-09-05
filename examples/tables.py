from heracles_evaluation.experiment_definition import (
    ExperimentConfiguration,
    ExperimentDescription,
)
import yaml
from heracles_evaluation.provider_integrations.openai.openai_client import (
    OpenaiClientConfig,
)
from heracles_evaluation.llm_agent import ModelInfo, LlmAgent, AgentInfo
from heracles_evaluation.prompt import PromptSettings

from heracles_evaluation.experiment_definition import PipelineRegistry
from heracles_evaluation.dsg_interfaces import (
    HeraclesDsgInterface,
    NoDsgInterface,
    PythonDsgInterface,
    InContextDsgInterfaceConfig,
)
import os

import feedforward_cypher_pipeline  # NOQA
import feedforward_in_context  # NOQA
import agentic_pipeline  # NOQA

# Method sweep table

# +---------------------+-------+-------+-------+-------+-------+-------+
# |                     |      Q&A      |      PDDL     |     Update    |
# |     Method          |   S   |   L   |   S   |   L   |   S   |   L   |
# +---------------------+-------+-------+-------+-------+-------+-------+
# | Cypher              |   x   |   -   |   x   |   -   |   x   |   x   |
# | Cypher (A)          |   x   |   -   |   x   |   -   |   x   |   x   |
# | Context Window      |   x   |   -   |   x   |   -   |   x   |   x   |
# | Context Window (A)  |   x   |   x   |   x   |   x   |   x   |   x   |
# | Python              |   x   |   x   |   x   |   x   |   x   |   x   |
# | Python (A)          |   x   |   x   |   x   |   x   |   x   |   x   |
# +---------------------+-------+-------+-------+-------+-------+-------+
# GPT-4.1

prompt_base = "prompts"
output_base = "tables/table1/configurations"
if not os.path.exists(output_base):
    print(f"{output_base} does not exist, creating it!")
    os.makedirs(output_base)

dsg_paths = {}
dsg_paths["small"] = ""
dsg_paths["westpoint"] = (
    "$HERACLES_EVALUATION_PATH/examples/scene_graphs/west_point_fused_map_wregions_labelspace.json"
)

# dsg_tags = ["small", "westpoint"]
dsg_tags = ["westpoint"]

methods = [
    "cypher",
    "agentic_cypher",
    "in_context",
    "agentic_in_context",
    "python",
    "agentic_python",
]
tasks = ["qa", "pddl", "update_qa"]


pipelines = [
    "agentic",
    "feedforward_cypher",
    "feedforward_in_context",
    "feedforward_python",
]

pr = PipelineRegistry.pipelines
method_to_pipeline = {
    "cypher": pr["feedforward_cypher"],
    "agentic_cypher": pr["agentic"],
    "in_context": pr["feedforward_in_context"],
    "agentic_in_context": None,  # pr["agentic_in_context"]
    "python": None,  # pr["feedforward_python"],
    "agentic_python": None,  # pr["agentic"],
}


def get_cypher_tool():
    return {
        "name": "run_cypher_query",
        "bound_args": {
            "dsgdb_conf": {
                "dsg_interface_type": "heracles",
                "uri": "neo4j://127.0.0.1:7687",
            }
        },
    }


def get_pipeline(method):
    return method_to_pipeline[method]


def get_questions_fn(task, dsg_tag):
    questions_base = "$HERACLES_EVALUATION_PATH/src/heracles_evaluation/questions"
    return f"{questions_base}/{dsg_tag}/{task}_questions.yaml"


def get_interface_for_method(m, dsg_tag):
    match m:
        case "cypher":
            return HeraclesDsgInterface(
                dsg_interface_type="heracles", uri="neo4j://127.0.0.1:7687"
            )
        case "python":
            return PythonDsgInterface(dsg_interface_type="python")
        case "in_context":
            return InContextDsgInterfaceConfig(
                dsg_interface_type="in_context", dsg_filepath=dsg_paths[dsg_tag]
            )
        case _:
            return NoDsgInterface(dsg_interface_type="none")


def get_cypher_qa_info(tool_interface):
    prompt_settings = PromptSettings(
        base_prompt=f"{prompt_base}/dsg_feedforward_prompt.yaml",
    )
    cypher_agent_info = AgentInfo(
        tool_interface=tool_interface,
        max_iterations=1,
        tools=[],
        prompt_settings=prompt_settings,
    )

    refine_prompt_settings = PromptSettings(
        base_prompt=f"{prompt_base}/dsg_feedforward_refinement_prompt.yaml",
        output_type="SLDP",
        sldp_answer_type_hint=True,
    )
    refine_agent_info = AgentInfo(
        tool_interface=tool_interface,
        max_iterations=1,
        tools=[],
        prompt_settings=refine_prompt_settings,
    )

    return {"generate-cypher": cypher_agent_info, "refine": refine_agent_info}


def get_cypher_pddl_info(tool_interface):
    prompt_settings = PromptSettings(
        base_prompt=f"{prompt_base}/pddl_feedforward_system_prompt.yaml",
    )
    cypher_agent_info = AgentInfo(
        tool_interface=tool_interface,
        max_iterations=1,
        tools=[],
        prompt_settings=prompt_settings,
    )

    refine_prompt_settings = PromptSettings(
        base_prompt=f"{prompt_base}/pddl_feedforward_refinement_system_prompt.yaml",
        output_type="PDDL",
    )
    refine_agent_info = AgentInfo(
        tool_interface=tool_interface,
        max_iterations=1,
        tools=[],
        prompt_settings=refine_prompt_settings,
    )

    return {"generate-cypher": cypher_agent_info, "refine": refine_agent_info}


def get_agentic_cypher_pddl_info(tool_interface):
    prompt_settings = PromptSettings(
        base_prompt=f"{prompt_base}/pddl_agent_system_prompt.yaml",
        output_type="PDDL",
    )
    agent = AgentInfo(
        tool_interface=tool_interface,
        max_iterations=6,
        tools=[get_cypher_tool()],
        prompt_settings=prompt_settings,
    )

    return {"main": agent}


def get_agentic_cypher_qa_info(tool_interface):
    prompt_settings = PromptSettings(
        base_prompt=f"{prompt_base}/agent_system_prompt_full_info.yaml",
        output_type="SLDP",
        sldp_answer_type_hint=True,
    )
    agent = AgentInfo(
        tool_interface=tool_interface,
        max_iterations=6,
        tools=[get_cypher_tool()],
        prompt_settings=prompt_settings,
    )

    return {"main": agent}


def get_in_context_qa_info(tool_interface):
    prompt_settings = PromptSettings(
        base_prompt=f"{prompt_base}/incontext_dsg_prompt.yaml",
        output_type="SLDP",
        sldp_answer_type_hint=True,
    )
    main = AgentInfo(
        tool_interface=tool_interface,
        max_iterations=1,
        tools=[],
        prompt_settings=prompt_settings,
    )

    return {"main": main}


def get_agentic_in_context_qa_info(tool_interface):
    raise NotImplementedError()


def get_agentic_in_context_pddl_info(tool_interface):
    raise NotImplementedError()


def get_in_context_pddl_info(tool_interface):
    prompt_settings = PromptSettings(
        base_prompt=f"{prompt_base}/incontext_pddl_dsg_prompt.yaml",
        output_type="PDDL",
    )
    main = AgentInfo(
        tool_interface=tool_interface,
        max_iterations=1,
        tools=[],
        prompt_settings=prompt_settings,
    )

    return {"main": main}


def get_python_pddl_info(tool_interface):
    raise NotImplementedError()


def get_python_qa_info(tool_interface):
    raise NotImplementedError()


def get_agentic_python_qa_info(tool_interface):
    raise NotImplementedError()


def get_agentic_python_pddl_info(tool_interface):
    raise NotImplementedError()


def get_agent_info_builder(method, task):
    # Need different tools depending on Method x Task
    # I guess you could imagine always giving both tools to the LLM
    # and then not requiring switching on task here.
    if method == "cypher" and task == "qa":
        return get_cypher_qa_info
    elif method == "cypher" and task == "pddl":
        return get_cypher_pddl_info
    elif method == "agentic_cypher" and task == "qa":
        return get_agentic_cypher_qa_info
    elif method == "agentic_cypher" and task == "pddl":
        return get_agentic_cypher_pddl_info

    elif method == "in_context" and task == "qa":
        return get_in_context_qa_info
    elif method == "in_context" and task == "pddl":
        return get_in_context_pddl_info
    elif method == "agentic_in_context" and task == "qa":
        return get_agentic_in_context_qa_info
    elif method == "agentic_in_context" and task == "pddl":
        return get_agentic_in_context_pddl_info

    elif method == "python" and task == "qa":
        return get_python_qa_info
    elif method == "python" and task == "pddl":
        return get_python_pddl_info
    elif method == "agentic_python" and task == "qa":
        return get_agentic_python_qa_info
    elif method == "agentic_python" and task == "pddl":
        return get_agentic_python_pddl_info
    else:
        raise NotImplementedError(
            f"Don't know to to construct agent_info for method {method}, task {task}"
        )


def get_agent_infos(method, task, tool_interface):
    return get_agent_info_builder(method, task)(tool_interface)


def get_phases(method, task):
    client = OpenaiClientConfig(client_type="openai", timeout=120)
    model_info = ModelInfo(model="gpt-5-nano")

    agent_infos = get_agent_infos(method, task, "openai")

    phases = {}
    for name, agent_info in agent_infos.items():
        phases[name] = LlmAgent(
            client=client, model_info=model_info, agent_info=agent_info
        )
    return phases


for m in methods:
    for t in tasks:
        for tag in dsg_tags:
            config_name = f"{m}_{t}_{tag}"
            print("config_name: ", config_name)
            configurations = {}
            metadata = {"config_name": config_name}
            try:
                config = ExperimentConfiguration(
                    pipeline=get_pipeline(m),
                    phases=get_phases(m, t),
                    dsg_interface=get_interface_for_method(m, tag),
                    questions=get_questions_fn(t, tag),
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

            exp = ExperimentDescription(
                metadata=metadata,
                configurations=configurations,
            )

            with open(f"{output_base}/{config_name}.yaml", "w") as fo:
                fo.write(yaml.dump(exp.model_dump()))
