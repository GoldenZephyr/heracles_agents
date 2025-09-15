# ruff: noqa: F811
import copy
import json
import logging
from typing import Callable

import tiktoken
from openai.types.responses.response import Response
from openai.types.responses.response_custom_tool_call import ResponseCustomToolCall
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_reasoning_item import ResponseReasoningItem
from plum import dispatch

from heracles_evaluation.agent_functions import (
    call_custom_tool_from_string,
    extract_tag,
)
from heracles_evaluation.llm_agent import LlmAgent
from heracles_evaluation.prompt import Prompt
from heracles_evaluation.provider_integrations.openai.openai_client import (
    OpenaiClientConfig,
)

logger = logging.getLogger(__name__)


@dispatch
def generate_prompt_for_agent(prompt: Prompt, agent: LlmAgent[OpenaiClientConfig]):
    p = copy.deepcopy(prompt)
    if agent.agent_info.tool_interface == "custom":
        # TODO: centralize custom tool prompt logic
        tool_command = "The following tools can be used to help formulate your answer. To call a tool, response with the function name and arguments between a tool tag, like this: <tool> my_function(arg1=1,arg2=2,arg3='3') </tool>.\n"
        for tool in agent.agent_info.tools.values():
            d = tool.to_custom()
            tool_command += d
        p.tool_description = tool_command
    return p.to_openai_json()


@dispatch
def iterate_messages(agent: LlmAgent[OpenaiClientConfig], messages: Response):
    for m in messages.output:
        yield m


@dispatch
def is_function_call(agent: LlmAgent[OpenaiClientConfig], message):
    """is_function_call should return true for messages that can be passed to call_function below"""
    return (
        isinstance(message, ResponseFunctionToolCall)
        and message.type == "function_call"
    )


@dispatch
def call_function(
    agent: LlmAgent[OpenaiClientConfig], tool_message: ResponseFunctionToolCall
):
    available_tools = agent.agent_info.tools
    name = tool_message.name
    args = json.loads(tool_message.arguments)
    # TODO: verify legal tool name
    logger.debug(f"Calling function {name} {args}")
    result = available_tools[name].function(**args)
    logger.debug(f"Result {result}")
    return result


@dispatch
def call_function(
    agent: LlmAgent[OpenaiClientConfig], tool_message: ResponseOutputMessage
):
    available_tools = agent.agent_info.tools
    tool_string = extract_tag("tool", tool_message.content[0].text)
    return call_custom_tool_from_string(available_tools, tool_string)


@dispatch
def make_tool_response(
    agent: LlmAgent[OpenaiClientConfig],
    tool_call_message: ResponseFunctionToolCall,
    result,
):
    m = {
        "type": "function_call_output",
        "call_id": tool_call_message.call_id,
        "output": str(result),
    }
    return m


@dispatch
def make_tool_response(
    agent: LlmAgent[OpenaiClientConfig],
    tool_call_message: ResponseOutputMessage,
    result,
):
    m = {"role": "user", "content": f"Output of tool call: {result}"}
    return m


@dispatch
def generate_update_for_history(
    agent: LlmAgent[OpenaiClientConfig], response: Response
) -> list:
    return response.output


@dispatch
def extract_answer(
    agent: LlmAgent[OpenaiClientConfig],
    extractor: Callable,
    message: ResponseOutputMessage,
):
    return extractor(message.content[0].text)


@dispatch
def get_text_body(response: Response):
    return "\n".join([get_text_body(m) for m in response.output])


@dispatch
def get_text_body(message: ResponseOutputMessage):
    return "\n".join([c.text for c in message.content])


@dispatch
def get_text_body(tool_call: ResponseFunctionToolCall):
    return f"{tool_call.name}({tool_call.arguments})"


@dispatch
def get_text_body(message: ResponseReasoningItem):
    if message.content is None:
        return ""
    return "\n".join(c.text for c in message.content)


@dispatch
def get_text_body(tool_call: ResponseCustomToolCall):
    return f"{tool_call.name}({tool_call.input})"


@dispatch
def count_message_tokens(agent: LlmAgent[OpenaiClientConfig], message: dict):
    model_name = agent.model_info.model
    if model_name == "gpt-5":
        model_name = "gpt-5-latest"  # tiktoken is broken
    enc = tiktoken.encoding_for_model(model_name)
    # https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
    num_tokens = 3
    for key, value in message.items():
        num_tokens += len(enc.encode(value))
    if key == "name":
        num_tokens += 1
    return num_tokens
