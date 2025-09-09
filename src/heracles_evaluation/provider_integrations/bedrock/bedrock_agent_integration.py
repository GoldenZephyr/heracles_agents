# ruff: noqa: F811, F401
import json
from typing import Callable

import tiktoken

from plum import dispatch

from heracles_evaluation.llm_agent import LlmAgent
from heracles_evaluation.prompt import Prompt
from heracles_evaluation.provider_integrations.bedrock.bedrock_client import (
    BedrockClientConfig,
)
import copy

from heracles_evaluation.agent_functions import (
    call_custom_tool_from_string,
    extract_tag,
)


@dispatch
def generate_prompt_for_agent(prompt: Prompt, agent: LlmAgent[BedrockClientConfig]):
    print("prompt for agent!")
    print("Tool interface: ", agent.agent_info.tool_interface)
    p = copy.deepcopy(prompt)

    if agent.agent_info.tool_interface == "custom":
        # TODO: centralize custom tool prompt logic
        tool_command = "The following tools can be used to help formulate your answer. To call a tool, response with the function name and arguments between a tool tag, like this: <tool> my_function(arg1=1,arg2=2,arg3='3') </tool>.\n"
        for tool in agent.agent_info.tools.values():
            d = tool.to_custom()
            tool_command += d
        p.tool_description = tool_command
    return p.to_bedrock_json()


@dispatch
def iterate_messages(agent: LlmAgent[BedrockClientConfig], response_dict: dict):
    for m in response_dict["output"]["message"]["content"]:
        yield m


@dispatch
def is_function_call(agent: LlmAgent[BedrockClientConfig], message):
    """is_function_call should return true for messages that can be passed to call_function below"""
    return False
    # return (
    #    isinstance(message, ResponseFunctionToolCall)
    #    and message.type == "function_call"
    # )


@dispatch
def call_function(agent: LlmAgent[BedrockClientConfig], tool_message: dict):
    print("tool_message: ", tool_message)
    available_tools = agent.agent_info.tools
    tool_string = extract_tag("tool", tool_message["text"])
    return call_custom_tool_from_string(available_tools, tool_string)


@dispatch
def make_tool_response(
    agent: LlmAgent[BedrockClientConfig],
    tool_call_message: dict,
    result,
):
    m = {"role": "user", "content": [{"text": f"Output of tool call: {result}"}]}
    return m


@dispatch
def generate_update_for_history(
    agent: LlmAgent[BedrockClientConfig], response: dict
) -> list:
    return response["output"]["message"]


@dispatch
def extract_answer(
    agent: LlmAgent[BedrockClientConfig],
    extractor: Callable,
    message: dict,
):
    return extractor(message["content"][0]["text"])


#
#
# @dispatch
# def get_text_body(response: Response):
#    return "\n".join([get_text_body(m) for m in response.output])
#
#
# @dispatch
# def get_text_body(message: ResponseOutputMessage):
#    return "\n".join([c.text for c in message.content])
#
#
# @dispatch
# def get_text_body(tool_call: ResponseFunctionToolCall):
#    return f"{tool_call.name}({tool_call.arguments})"
#
#
# @dispatch
# def get_text_body(message: ResponseReasoningItem):
#    if message.content is None:
#        return None
#    return "\n".join(c.text for c in message.content)
#
#
# @dispatch
# def get_text_body(tool_call: ResponseCustomToolCall):
#    return f"{tool_call.name}({tool_call.input})"
#
#


@dispatch
def count_message_tokens(agent: LlmAgent[BedrockClientConfig], message: dict):
    enc = tiktoken.get_encoding("cl100k_base")

    if "content" in message:
        # when we sent a message
        num_tokens = 3
        for block in message["content"]:
            for key, value in block.items():
                num_tokens += len(enc.encode(value))
        return num_tokens
    elif "text" in message:
        return len(enc.encode(message["text"]))
    elif "message" in message:
        return len(
            enc.encode(" ".join([c["text"] for c in message["message"]["content"]]))
        )
    else:
        raise NotImplementedError("Not sure how to process message: ", message)
