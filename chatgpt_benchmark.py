#!/usr/bin/env python3
import openai
import os
import yaml
from heracles.prompt_schema import Prompt

from heracles.query_interface import Neo4jWrapper


from model_info import ModelInfo

from dataclasses import asdict
from rich.progress import track

import sys

key = os.getenv("DSG_OPENAI_API_KEY")

if len(sys.argv) < 3:
    print("Usage: ./chatgpt_benchmark.py question_set_name output_suffix")
    exit(1)
question_set_name = sys.argv[1]
output_suffix = sys.argv[2]

# question_set_name = "nina_questions"
question_dir = "yaml_pipeline/question_sets"
question_set = f"{question_set_name}.yaml"

question_path = os.path.join(question_dir, question_set)

with open(question_path, "r") as fo:
    eval_questions = yaml.safe_load(fo)


client = openai.OpenAI(
    api_key=key,
    timeout=10,
)


with open("single_query_full_info_prompt.yaml", "r") as fo:
    prompt_yaml = yaml.safe_load(fo)

prompt_obj = Prompt.from_dict(prompt_yaml)
print("Base prompt: ", prompt_obj)


# IP / Port for database
URI = "neo4j://127.0.0.1:7687"
# Database name / password for database
AUTH = ("neo4j", "neo4j_pw")

# Assumes that the scene graph has already been loaded into the database
with Neo4jWrapper(URI, AUTH, atomic_queries=True, print_profiles=False) as db:
    objects = db.query(
        "MATCH (n: Object) RETURN DISTINCT n.class as class, COUNT(*) as count"
    )
    print("objects:")
    print(objects)

    model_info = ModelInfo(model="gpt-4.1-nano", temperature=0.2, seed=100)
    # model="gpt-4o-mini",
    # model="gpt-4o",
    eval_questions["answer_model_metadata"] = asdict(model_info)

    for q in track(eval_questions["questions"], description="Processing..."):
        print(f'\nAsking question called {q["name"]}...\n')

        prompt = prompt_obj.to_openai_json(q["question"])
        response = client.chat.completions.create(
            model=model_info.model,
            messages=prompt,
            temperature=model_info.temperature,
            seed=model_info.seed,
            response_format={"type": "json_object"},
        )
        print("response: ", response)
        cypher_text = response.choices[0].message.content
        print("\n cypher query: ", cypher_text)
        cypher_dict = eval(cypher_text)
        cypher_query = cypher_dict["cypher"]

        try:
            query_result = str(db.query(cypher_query))
            valid_cypher_query = True
        except Exception as ex:
            print(ex)
            query_result = str(ex)
            valid_cypher_query = False

        q["cot"] = cypher_dict["chain of thought"]
        q["cypher"] = cypher_query
        q["valid_cypher"] = valid_cypher_query
        q["answer"] = query_result

output_eval_results = os.path.join(
    "yaml_pipeline",
    "intermediate_answers",
    f"{question_set_name}_answers" + output_suffix + ".yaml",
)
with open(output_eval_results, "w") as fo:
    fo.write(yaml.dump(eval_questions, sort_keys=False))
