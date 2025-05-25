import openai
import os
import yaml

from sldp_lang import extract_uniform_keys, sldp_equals, get_sldp_type, parse_sldp

from answer_comparators import (
    convert_to_set,
    convert_to_number,
    convert_to_list,
    convert_to_dict,
    convert_to_set_of_dicts,
)
from model_info import ModelInfo
from dataclasses import asdict


key = os.getenv("DSG_OPENAI_API_KEY")

client = openai.OpenAI(
    api_key=key,
    timeout=10,
)

with open("temp_out.yaml", "r") as fo:
    result_yaml = yaml.safe_load(fo)

for problem in result_yaml["questions"]:
    print(f'\Post-processing question called {problem["name"]}...\n')

    model_info = ModelInfo(model="gpt-4.1-nano", temperature=0.01, seed=100)

    result_yaml["refinement_model_metadata"] = asdict(model_info)

    def refinement_fn(p):
        message = [{"role": "developer", "content": p}]
        r = client.chat.completions.create(
            model=model_info.model,
            messages=message,
            temperature=model_info.temperature,
            seed=model_info.seed,
        )
        return r.choices[0].message.content

    solution_type = get_sldp_type(problem["solution"])
    refinement_type = problem["refinement_type"]
    if solution_type != refinement_type:
        print(
            f"WARNING: Solution type {solution_type} but refinement type given as {refinement_type}"
        )
    match refinement_type:
        case "set":
            refined_answer = convert_to_set(
                refinement_fn, problem["question"], problem["answer"]
            )
        case "number":
            refined_answer = convert_to_number(
                refinement_fn, problem["question"], problem["answer"]
            )
        case "list":
            refined_answer = convert_to_list(
                refinement_fn, problem["question"], problem["answer"]
            )
        case "dict":
            refined_answer = convert_to_dict(
                refinement_fn, problem["question"], problem["answer"]
            )
        case "set_of_dicts":
            keys = extract_uniform_keys(problem["solution"])
            refined_answer = convert_to_set_of_dicts(
                refinement_fn, problem["question"], problem["answer"], keys=keys
            )

    try:
        parse_sldp(refined_answer)
        valid_sldp = True
    except Exception:
        valid_sldp = False

    if valid_sldp:
        correct = sldp_equals(problem["solution"], refined_answer)
    else:
        correct = False
    print("Correct: ", correct)

    problem["sldp_output"] = refined_answer
    problem["valid_sldp"] = valid_sldp
    problem["correct"] = correct

with open("refined_out.yaml", "w") as fo:
    fo.write(yaml.dump(result_yaml, sort_keys=False))
