# ruff: noqa: F811
"""Agent integration for the OpenAI Chat Completions client.

Provides plum dispatch overloads so AgentContext can drive an LlmAgent
backed by OpenaiCompletionsClientConfig (Chat Completions API).
"""

import copy
import json
import logging
from typing import Callable

import tiktoken
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from plum import dispatch

from heracles_agents.agent_functions import (
    call_custom_tool_from_string,
    extract_tag,
)
from heracles_agents.llm_agent import LlmAgent
from heracles_agents.prompt import Prompt
from heracles_agents.provider_integrations.openai.openai_completions_client import (
    OpenaiCompletionsClientConfig,
)

logger = logging.getLogger(__name__)


# ── prompt ───────────────────────────────────────────────────────────────

@dispatch
def generate_prompt_for_agent(
    prompt: Prompt, agent: LlmAgent[OpenaiCompletionsClientConfig]
):
    p = copy.deepcopy(prompt)
    if agent.agent_info.tool_interface == "custom":
        tool_command = (
            "The following tools can be used to help formulate your answer. "
            "To call a tool, respond with the function name and arguments between "
            "a tool tag, like this: <tool> my_function(arg1=1,arg2=2,arg3='3') </tool>.\n"
        )
        for tool in agent.agent_info.tools.values():
            d = tool.to_custom()
            tool_command += d
        p.tool_description = tool_command
    return p.to_openai_json()


# ── iterate messages ─────────────────────────────────────────────────────

@dispatch
def iterate_messages(
    agent: LlmAgent[OpenaiCompletionsClientConfig], messages: ChatCompletion
):
    message = messages.choices[0].message
    yield message
    if message.tool_calls:
        for tc in message.tool_calls:
            yield tc


# ── function call detection ─────────────────────────────────────────────

@dispatch
def is_function_call(
    agent: LlmAgent[OpenaiCompletionsClientConfig],
    message: ChatCompletionMessageToolCall,
):
    return message.type == "function"


@dispatch
def is_function_call(
    agent: LlmAgent[OpenaiCompletionsClientConfig],
    message: ChatCompletionMessage,
):
    return False


# ── call function ────────────────────────────────────────────────────────

@dispatch
def call_function(
    agent: LlmAgent[OpenaiCompletionsClientConfig],
    tool_message: ChatCompletionMessageToolCall,
):
    available_tools = agent.agent_info.tools
    name = tool_message.function.name
    args = json.loads(tool_message.function.arguments)
    logger.debug(f"Calling function {name} {args}")
    result = available_tools[name].function(**args)
    logger.debug(f"Result {result}")
    return result


@dispatch
def call_function(
    agent: LlmAgent[OpenaiCompletionsClientConfig],
    tool_message: ChatCompletionMessage,
):
    available_tools = agent.agent_info.tools
    tool_string = extract_tag("tool", tool_message.content)
    return call_custom_tool_from_string(available_tools, tool_string)


# ── tool responses ───────────────────────────────────────────────────────

@dispatch
def make_tool_response(
    agent: LlmAgent[OpenaiCompletionsClientConfig],
    tool_call_message: ChatCompletionMessageToolCall,
    result,
):
    return {
        "role": "tool",
        "tool_call_id": tool_call_message.id,
        "content": str(result),
    }


@dispatch
def make_tool_response(
    agent: LlmAgent[OpenaiCompletionsClientConfig],
    tool_call_message: ChatCompletionMessage,
    result,
):
    return {"role": "user", "content": f"Output of tool call: {result}"}


# ── history ──────────────────────────────────────────────────────────────

@dispatch
def generate_update_for_history(
    agent: LlmAgent[OpenaiCompletionsClientConfig], response: ChatCompletion
) -> list:
    message = response.choices[0].message
    entry = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        entry["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]
    return [entry]


# ── answer extraction ────────────────────────────────────────────────────

@dispatch
def extract_answer(
    agent: LlmAgent[OpenaiCompletionsClientConfig],
    extractor: Callable,
    message: ChatCompletionMessage,
):
    if message.content:
        return extractor(message.content)
    return None


@dispatch
def extract_answer(
    agent: LlmAgent[OpenaiCompletionsClientConfig],
    extractor: Callable,
    message: dict,
):
    content = message.get("content")
    if content:
        return extractor(content)
    return None


# ── text body extraction ─────────────────────────────────────────────────

@dispatch
def get_text_body(response: ChatCompletion):
    return get_text_body(response.choices[0].message)


@dispatch
def get_text_body(message: ChatCompletionMessage):
    parts = []
    if message.content:
        parts.append(message.content)
    if message.tool_calls:
        for tc in message.tool_calls:
            parts.append(f"{tc.function.name}({tc.function.arguments})")
    return "\n".join(parts) if parts else ""


@dispatch
def get_text_body(tool_call: ChatCompletionMessageToolCall):
    return f"{tool_call.function.name}({tool_call.function.arguments})"


# ── token counting ───────────────────────────────────────────────────────

@dispatch
def count_message_tokens(
    agent: LlmAgent[OpenaiCompletionsClientConfig], message: ChatCompletionMessage
):
    try:
        enc = tiktoken.encoding_for_model(agent.model_info.model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    text = message.content or ""
    return len(enc.encode(text))


@dispatch
def count_message_tokens(
    agent: LlmAgent[OpenaiCompletionsClientConfig],
    message: ChatCompletionMessageToolCall,
):
    try:
        enc = tiktoken.encoding_for_model(agent.model_info.model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    text = f"{message.function.name}({message.function.arguments})"
    return len(enc.encode(text))


@dispatch
def count_message_tokens(
    agent: LlmAgent[OpenaiCompletionsClientConfig], message: dict
):
    try:
        enc = tiktoken.encoding_for_model(agent.model_info.model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    num_tokens = 3
    for key, value in message.items():
        if isinstance(value, str):
            num_tokens += len(enc.encode(value))
    return num_tokens
