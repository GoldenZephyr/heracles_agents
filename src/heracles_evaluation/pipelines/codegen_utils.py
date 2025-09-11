import logging
import math
from typing import Any, Dict

import spark_dsg
import yaml

logger = logging.getLogger(__name__)


def load_dsg(dsg_filepath, label_path=None):
    """
    Loads a Spark Dynamic Scene Graph (DSG) from the specified file path and optionally augments it with label metadata.
    Args:
        dsg_filepath (str or Path): Path to the DSG file to load.
        label_path (str or Path, optional): Path to a YAML file containing label definitions. If provided, label metadata
            will be added to the DSG's metadata.
    Returns:
        DynamicSceneGraph: The loaded DSG object, potentially augmented with label and layer metadata.
    """
    G = spark_dsg.DynamicSceneGraph.load(dsg_filepath)
    logger.info(f"DSG loaded from {dsg_filepath}")

    if label_path:
        logger.info(f"Loading labels from {label_path}")

        with open(str(label_path), "r") as fo:
            labelspace = yaml.safe_load(fo)
        id_to_label = {
            item["label"]: item["name"] for item in labelspace["label_names"]
        }
        G.metadata.add({"labelspace": id_to_label})
        region_ls = {
            0: "unknown",
            1: "road",
            2: "field",
            3: "shelter",
            4: "indoor",
            5: "stairs",
            6: "sidewalk",
            7: "path",
            8: "boundary",
            9: "shore",
            10: "ground",
            11: "dock",
            12: "parking",
            13: "footing",
        }
        G.metadata.add({"room_labelspace": region_ls})

        layers = {
            2: "Object",
            5: "Building",
            20: "MeshPlace",
            3: "Place",
            4: "Room",
        }

        G.metadata.add({"LayerIdToLayerStr": layers})
        logger.info(f"Labels loaded from {label_path}")
    return G


def load_dsg_api_prompt(
    api_prompt_file, include_descriptions: bool = False, include_examples: bool = False
) -> str:
    """
    Loads the API specification from a YAML file and formats it into a string prompt.

    Args:
        include_descriptions: If True, include descriptions for classes, methods, etc.
        include_examples: If True, include code examples.

    Returns:
        A formatted string representing the public API surface.
    """
    with open(api_prompt_file, "r") as f:
        prompt_yaml = yaml.safe_load(f)

    api_spec = prompt_yaml.get("api", {})
    prompt_parts = []

    # --- API Header ---
    api_name = api_spec.get("name", "API")
    api_version = api_spec.get("version", "")
    prompt_parts.append(f"# {api_name} API Reference (Version {api_version})\n")
    if include_descriptions and api_spec.get("description"):
        prompt_parts.append(f"{api_spec['description']}\n")

    # --- Process Classes In Spark DSG---
    for cls in api_spec.get("classes", []):
        class_name = cls.get("name")
        if not class_name:
            continue
        include_class = cls.get("include", False)
        if not include_class:
            continue

        prompt_parts.append(f"## class {class_name}:")
        if include_descriptions and cls.get("description"):
            prompt_parts.append(f'  """{cls["description"]}"""')

        # Constructor
        constructor_data = cls.get("constructor")
        if constructor_data and constructor_data.get("include") is True:
            prompt_parts.append(
                format_callable_api(
                    constructor_data,
                    class_name,
                    is_constructor=True,
                    include_descriptions=include_descriptions,
                    include_examples=include_examples,
                )
            )

        # Properties
        properties = cls.get("properties", [])
        if properties:
            prop_lines = ["\n  # Properties"]
            for prop in properties:
                prop_line = f"  {prop['name']}: {prop['type']}"
                if include_descriptions and prop.get("description"):
                    prop_line += f"  # {prop['description']}"
                prop_lines.append(prop_line)
            prompt_parts.append("\n".join(prop_lines))

        # Methods
        methods = [m for m in cls.get("methods", []) if m.get("include") is True]
        if methods:
            prompt_parts.append("\n  # Methods")
            for method in methods:
                prompt_parts.append(
                    format_callable_api(
                        method,
                        class_name,
                        is_constructor=False,
                        include_descriptions=include_descriptions,
                        include_examples=include_examples,
                    )
                )

        # Nested Enums
        enums = cls.get("enums", [])
        if enums:
            prompt_parts.append("\n  # Enums")
            for enum in enums:
                prompt_parts.append(f"  class {enum['name']}:")
                for val in enum.get("values", []):
                    prompt_parts.append(f"    {val} = ...")

        prompt_parts.append("\n" + "-" * 40 + "\n")  # Separator

    # --- Process Top-Level Enums ---
    enums = api_spec.get("enums", [])
    if enums:
        for enum in enums:
            prompt_parts.append(f"class {enum['name']}:")
            if include_descriptions and enum.get("description"):
                prompt_parts.append(f'  """{enum["description"]}"""')
            for val in enum.get("values", []):
                name = val if isinstance(val, str) else val.get("name")
                desc = ""
                if (
                    include_descriptions
                    and isinstance(val, dict)
                    and val.get("description")
                ):
                    desc = f"  # {val['description']}"
                prompt_parts.append(f"  {name} = ...{desc}")
            prompt_parts.append("")

    return "\n".join(prompt_parts)


def format_callable_api(
    data: Dict[str, Any],
    class_name: str,
    is_constructor: bool,
    include_descriptions: bool,
    include_examples: bool,
) -> str:
    """Helper to format a method or constructor into a string."""
    parts = []

    # --- Signature ---
    name = "__init__" if is_constructor else data.get("name", "")

    # Format inputs/parameters
    inputs = data.get("inputs", [])
    params = ["self"] if not data.get("is_static", False) else []
    for param in inputs:
        params.append(f"{param['name']}: {param['type']}")
    param_str = ", ".join(params)

    # Format output/return type
    output_str = ""
    if not is_constructor and "output" in data and data["output"]["type"] != "None":
        output_str = f" -> {data['output']['type']}"

    signature = f"  def {name}({param_str}){output_str}:"
    parts.append(signature)

    # --- Docstring ---
    if include_descriptions:
        description = data.get("description", "")
        docstring_parts = [f'    """{description}']

        if inputs:
            docstring_parts.append("\n    Args:")
            for param in inputs:
                param_desc = param.get("description", "")
                docstring_parts.append(
                    f"      {param['name']} ({param['type']}): {param_desc}"
                )

        if output_str:
            output_data = data.get("output", {})
            output_desc = output_data.get("description", "")
            docstring_parts.append("\n    Returns:")
            docstring_parts.append(f"      {output_data['type']}: {output_desc}")

        docstring_parts.append('    """')
        parts.append("\n".join(docstring_parts))

    # --- Example ---
    if include_examples and data.get("example"):
        example = data["example"].strip()
        # Indent the example block for clarity
        indented_example = "\n".join([f"    # {line}" for line in example.split("\n")])
        parts.append(f"    # Example:\n{indented_example}")

    return "\n".join(parts)


def execute_generated_code(python_code: str, scene_graph: spark_dsg.DynamicSceneGraph):
    # # Extract code between <python>...</python> tags if present
    # python_code_match = re.search(r"<python>(.*?)</python>", python_code, re.DOTALL)
    # if not python_code_match:
    #     error_msg = "warning, No <python>...</python> tags found in the generated code."
    #     print(error_msg)
    #     return error_msg
    # code_to_execute = python_code_match.group(1).strip()
    logger.info(f"Executing generated code:\n{python_code}")

    try:
        local_scope = {}
        exec_globals = {
            "spark_dsg": spark_dsg,
            "math": math,
        }  # make spark dsg and math available inside the execution blcok

        # Execute the code, which defines the 'solve_task' function in the local scope (generated code)
        exec(python_code, exec_globals, local_scope)
        solve_task_func = local_scope.get("solve_task")
        if callable(solve_task_func):
            logger.info("Executing function 'solve_task'...")
            result = solve_task_func(scene_graph)
            logger.info(f"Execution successful. Result: {result}")
            return True, result
        else:
            logger.info("'solve_task' function not found in the generated code.")
            return False, "'solve_task' function not found in the generated code."

    except Exception as e:
        logger.info(f"Error executing generated code: {e}", exc_info=True)
        return False, str(e)
