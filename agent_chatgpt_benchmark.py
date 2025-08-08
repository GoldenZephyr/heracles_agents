#!/usr/bin/env python3
import json
import os
import re
import sys
import time
from dataclasses import asdict

import openai
import yaml
from heracles.prompt_schema import Prompt
from heracles.query_interface import Neo4jWrapper
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from rich.progress import track
from sldp_lang import get_sldp_type

from answer_comparators import (
    agent_dict_answer_ps,
    agent_list_answer_ps,
    agent_number_answer_ps,
    agent_set_answer_ps,
    agent_string_answer_ps,
)
from model_info import ModelInfo


def extract_answer(string):
    matches = re.findall("<answer>([\s\S]*?)<\/answer>", string, re.MULTILINE)
    if len(matches) > 1:
        # TODO: eventually fail more gracefully?
        raise Exception("Found multiple answeres!")
    if len(matches) == 0:
        return None
    return matches[0]


def format_conversation(messages):
    conversation_string = ""

    for m in messages:
        if type(m) is ChatCompletionMessage:
            conversation_string += f"{m.role}: "
            if m.content is not None:
                conversation_string += m.content
            if m.tool_calls is not None:
                for t in m.tool_calls:
                    conversation_string += str(t.function) + "\n"

        else:
            role = m["role"]
            conversation_string += f"{role}: "
            if role == "tool":
                conversation_string += m["name"] + ": "
                conversation_string += m["content"] + "\n"
            elif m["content"] is not None:
                conversation_string += m["content"] + "\n"

    return conversation_string


key = os.getenv("DSG_OPENAI_API_KEY")

client = openai.OpenAI(
    api_key=key,
    timeout=10,
)

if len(sys.argv) < 3:
    print("Usage: ./agent_chatgpt_benchmark.py questions_path output_path")
    exit(1)

question_path = sys.argv[1]
output_eval_results = sys.argv[2]

start_index = 0
# start_index = 11

# with open("agent_system_prompt_full_info.yaml", "r") as fo:
with open("agent_system_prompt_no_info.yaml", "r") as fo:
    prompt_yaml = yaml.safe_load(fo)


with open(question_path, "r") as fo:
    eval_questions = yaml.safe_load(fo)

# IP / Port for database
URI = "neo4j://127.0.0.1:7687"
# Database name / password for database
AUTH = ("neo4j", "neo4j_pw")

# tools
cypher_tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_cypher_query",
            "description": "Execute a Cypher query against the Neo4j database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": [
                    "query",
                ],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]


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
    eval_questions["answer_model_metadata"]["agent"] = True

    debug_results = []
    for q in track(
        eval_questions["questions"][start_index:], description="Processing..."
    ):
        print(f"\nAsking question called {q['name']}...\n")

        solution_type = get_sldp_type(q["solution"])
        refinement_type = q["refinement_type"]
        if solution_type != refinement_type:
            print(
                f"WARNING: Solution type {solution_type} but refinement type given as {refinement_type}"
            )

        prompt_obj = Prompt.from_dict(prompt_yaml)
        match refinement_type:
            case "set":
                prompt_obj.novel_instruction_ps += agent_set_answer_ps
            case "number":
                prompt_obj.novel_instruction_ps += agent_number_answer_ps
            case "list":
                prompt_obj.novel_instruction_ps += agent_list_answer_ps
            case "dict":
                prompt_obj.novel_instruction_ps += agent_dict_answer_ps
            case "set_of_dicts":
                raise NotImplementedError()
            case "string":
                prompt_obj.novel_instruction_ps += agent_string_answer_ps

        messages = prompt_obj.to_openai_json(q["question"])

        # TODO: for now, so the agent doesn't take over the world
        max_iterations = 10
        for i in range(max_iterations):
            print(f"Iteration {i + 1} of {max_iterations}")
            print("messages: ", messages)

            got_response = False
            n_retries = 3
            for n in range(n_retries):
                try:
                    response = client.chat.completions.create(
                        model=model_info.model,
                        messages=messages,
                        temperature=model_info.temperature,
                        seed=model_info.seed,
                        response_format={"type": "text"},
                        tools=cypher_tools,
                    )
                    got_response = True
                except openai.APITimeoutError as ex:
                    print("API Request timed out!")
                    print(ex)
                    print("Waiting for 60 seconds...")
                    time.sleep(60)
                except openai.RateLimitError as ex:
                    print(ex)
                    print("Rate Limit Hit!. Waiting 60s")
                    time.sleep(60)
            if not got_response:
                print("Didn't get openai response!")
                continue

            print("response: ", response)
            # check if the response is a tool call
            all_notifications = []
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                # tool calls need the llm response message appended
                messages.append(response.choices[0].message)

                for tool in tool_calls:
                    tool_call_id = tool.id
                    tool_function_name = tool.function.name
                    tool_query_string = json.loads(tool.function.arguments)["query"]
                    if tool_function_name == "execute_cypher_query":
                        print("\n cypher query: ", tool_query_string)
                        try:
                            result, notifications = db.query_with_notifications(
                                tool_query_string
                            )
                            query_result = str(result)
                            if len(query_result) > 10000:
                                query_result = (
                                    query_result[:10000] + "\n < RESULTS TRUNCATED >"
                                )
                        except Exception as ex:
                            print(ex)
                            query_result = str(ex)
                            notifications = None

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": tool_function_name,
                                "content": query_result,
                            }
                        )
                        if notifications is not None:
                            useful_notifications = [
                                n
                                for n in notifications
                                if n["severity"] != "INFORMATION"
                            ]
                            all_notifications.append(useful_notifications)
                    else:
                        # TODO: probably shouldn't raise here, but for development we want to see if this ever happens
                        raise Exception(
                            "Received unknown tool call {tool_function_name}"
                        )

                # Need to append notifications message after *all* tool call responses
                if len(all_notifications) > 0:
                    messages.append(
                        {
                            "role": "developer",
                            "content": str(all_notifications),
                        }
                    )

            else:
                # no tool call, do post-processing and break
                # We expect that the final output is formatted as a dict with a "chain of thought" key and a final_answer key
                # TODO: check if we stopped for some other reason, like running into token limit?
                messages.append(response.choices[0].message)
                response_text = response.choices[0].message.content

                answer = extract_answer(response_text)

                # response_dict = eval(response_text)
                # if "final_answer" in response_dict:
                #    answer = str(response_dict["final_answer"])
                # else:
                #    answer = None
                q["answer"] = answer
                break

        # store debugging results per step
        q["messages"] = format_conversation(messages)


with open(output_eval_results, "w") as fo:
    fo.write(yaml.dump(eval_questions, sort_keys=False))
