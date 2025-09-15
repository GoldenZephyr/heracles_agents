# ruff: noqa: F811, F401
import copy
import json
from typing import Callable

import tiktoken
from plum import dispatch

from heracles_evaluation.agent_functions import (
    call_custom_tool_from_string,
    extract_tag,
)
from heracles_evaluation.llm_agent import LlmAgent
from heracles_evaluation.prompt import Prompt
from heracles_evaluation.provider_integrations.bedrock.bedrock_client import (
    BedrockClientConfig,
)


@dispatch
def generate_prompt_for_agent(prompt: Prompt, agent: LlmAgent[BedrockClientConfig]):
    p = copy.deepcopy(prompt)

    if agent.agent_info.tool_interface == "custom":
        # TODO: centralize custom tool prompt logic
        tool_command = "The following tools can be used to help formulate your answer. To call a tool, response with the function name and arguments between a tool tag, like this: <tool> my_function(arg1=1,arg2=2,arg3='3') </tool>. Use a single tool call per message. Do not <answer> and tool call in the same message\n"
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
def is_function_call(agent: LlmAgent[BedrockClientConfig], message: dict):
    """is_function_call should return true for messages that can be passed to call_function below"""
    print("bedrock is_function_call message: ", message)
    if "toolUse" in message:
        return True
    return False


@dispatch
def call_function(agent: LlmAgent[BedrockClientConfig], tool_message: dict):
    available_tools = agent.agent_info.tools
    if "text" in tool_message:
        tool_string = extract_tag("tool", tool_message["text"])
        return call_custom_tool_from_string(available_tools, tool_string)
    elif "toolUse" in tool_message:
        name = tool_message["toolUse"]["name"]
        # args = json.loads(tool_message["toolUse"]["input"])
        args = tool_message["toolUse"]["input"]
        # TODO: verify legal tool name
        return available_tools[name].function(**args)
    else:
        raise NotImplementedError(
            f"Don't know how to call function from: {tool_message}"
        )


@dispatch
def make_tool_response(
    agent: LlmAgent[BedrockClientConfig],
    tool_call_message: dict,
    result,
):
    if "toolUse" in tool_call_message:
        if not isinstance(result, str):
            result = str(result)
        m = {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": tool_call_message["toolUse"]["toolUseId"],
                        "content": [{"text": result}],
                    }
                }
            ],
        }
    else:
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
    try:
        return extractor(message["content"][0]["text"])
    except:
        return ""


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
def count_message_tokens(agent: LlmAgent[BedrockClientConfig], message: str):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(message))


@dispatch
def count_message_tokens(agent: LlmAgent[BedrockClientConfig], message: dict):
    enc = tiktoken.get_encoding("cl100k_base")

    if "content" in message:
        # when we sent a message
        num_tokens = 3
        for block in message["content"]:
            for key, value in block.items():
                num_tokens += count_message_tokens(agent, value)
                # num_tokens += len(enc.encode(value))
        return num_tokens
    elif "text" in message:
        return len(enc.encode(message["text"]))
    elif "message" in message:
        return len(
            enc.encode(" ".join([c["text"] for c in message["message"]["content"]]))
        )
    elif "toolUse" in message:
        return count_message_tokens(agent, message["toolUse"])
    if "toolUseId" in message:
        total = len(enc.encode(message["name"]))
        for argname, argval in message["input"].items():
            total += len(enc.encode(argname))
            total += len(enc.encode(argval))
        return total
    else:
        raise NotImplementedError("Not sure how to process message: ", message)
