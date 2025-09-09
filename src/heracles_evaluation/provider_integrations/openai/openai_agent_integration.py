# ruff: noqa: F811
import json
from typing import Callable

import tiktoken
from openai.types.responses.response import Response
from openai.types.responses.response_custom_tool_call import ResponseCustomToolCall
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_reasoning_item import ResponseReasoningItem
from plum import dispatch

from heracles_evaluation.llm_agent import LlmAgent
from heracles_evaluation.prompt import Prompt
from heracles_evaluation.provider_integrations.openai.openai_client import (
    OpenaiClientConfig,
)


@dispatch
def generate_prompt_for_agent(prompt: Prompt, agent: LlmAgent[OpenaiClientConfig]):
    return prompt.to_openai_json()


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
    return available_tools[name].function(**args)


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
    enc = tiktoken.encoding_for_model(model_name)
    # https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
    num_tokens = 3
    for key, value in message.items():
        num_tokens += len(enc.encode(value))
    if key == "name":
        num_tokens += 1
    return num_tokens
