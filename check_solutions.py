#!/usr/bin/env python3
import yaml

from sldp_lang import sldp_equals, get_sldp_type, parse_sldp


from rich.progress import track
import sys

if len(sys.argv) < 2:
    print("Usage: ./check_solutions.py yaml_path")
    exit(1)

yaml_path = sys.argv[1]


with open(yaml_path, "r") as fo:
    result_yaml = yaml.safe_load(fo)

for problem in track(result_yaml["questions"], description="Refining..."):
    print(f'\nChecking question called {problem["name"]}...\n')

    solution_type = get_sldp_type(problem["solution"])

    if "answer" in problem:
        answer = problem["answer"]
        try:
            parse_sldp(answer)
            valid_sldp = True
        except Exception:
            valid_sldp = False
    else:
        answer = ""
        valid_sldp = False

    if valid_sldp:
        correct = sldp_equals(problem["solution"], answer)
    else:
        correct = False
    print("Correct: ", correct)

    problem["sldp_output"] = answer
    problem["valid_cypher"] = False
    problem["valid_sldp"] = valid_sldp
    problem["correct"] = correct


with open(yaml_path, "w") as fo:
    fo.write(yaml.dump(result_yaml, sort_keys=False))
