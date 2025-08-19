import json
from typing import Callable

from openai.types.responses.response import Response
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_output_message import ResponseOutputMessage
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
