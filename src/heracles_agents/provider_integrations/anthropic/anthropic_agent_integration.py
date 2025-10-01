# ruff: noqa: F811
import logging

import tiktoken
from anthropic import types as anthropic_types
from anthropic.types.message import Message
from anthropic.types.text_block import TextBlock
from anthropic.types.tool_use_block import ToolUseBlock
from plum import dispatch

from heracles_agents.agent_functions import (
    call_custom_tool_from_string,
    extract_tag,
)
from heracles_agents.llm_agent import LlmAgent
from heracles_agents.prompt import Prompt
from heracles_agents.provider_integrations.anthropic.anthropic_client import (
    AnthropicClientConfig,
)

logger = logging.getLogger(__name__)


@dispatch
def generate_prompt_for_agent(prompt: Prompt, agent: LlmAgent[AnthropicClientConfig]):
    return prompt.to_anthropic_json()


@dispatch
def is_function_call(agent: LlmAgent[AnthropicClientConfig], message):
    return isinstance(message, ToolUseBlock) and message.type == "tool_use"


@dispatch
def iterate_messages(agent: LlmAgent[AnthropicClientConfig], messages: Message):
    for m in messages.content:
        yield m


@dispatch
def call_function(agent: LlmAgent[AnthropicClientConfig], tool_message: ToolUseBlock):
    available_tools = agent.agent_info.tools
    name = tool_message.name
    args = tool_message.input
    # TODO: verify legal tool name
    return available_tools[name].function(**args)


@dispatch
def call_function(agent: LlmAgent[AnthropicClientConfig], tool_message: TextBlock):
    available_tools = agent.agent_info.tools
    tool_string = extract_tag("tool", tool_message.content)
    return call_custom_tool_from_string(available_tools, tool_string)


@dispatch
def make_tool_response(
    agent: LlmAgent[AnthropicClientConfig], tool_call_message: ToolUseBlock, result
):
    m = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_call_message.id,
                "content": str(result),
            }
        ],
    }
    return m


@dispatch
def make_tool_response(
    agent: LlmAgent[AnthropicClientConfig], tool_call_message: TextBlock, result
):
    m = {"role": "user", "content": f"Output of tool call: {result}"}
    return m


@dispatch
def generate_update_for_history(
    agent: LlmAgent[AnthropicClientConfig], response: Message
) -> list:
    m = anthropic_types.MessageParam(role="assistant", content=response.content)
    return [m]


@dispatch
def extract_answer(agent: LlmAgent[AnthropicClientConfig], extractor, message: dict):
    return extractor(message["content"][0].text)


def get_content_blocks_of_type(t, message):
    blocks = []
    for b in message.content:
        if b.type == t:
            blocks.append(b)
    return blocks


@dispatch
def get_text_body(message: Message):
    text_blocks = get_content_blocks_of_type("text", message)
    if len(text_blocks) > 1:
        logger.warning("Found multiple text blocks in message response")
    return "\n".join(b.text for b in text_blocks)


@dispatch
def get_text_body(message: ToolUseBlock):
    return f"{message.name}({message.input})"


@dispatch
def get_text_body(block: TextBlock):
    return block.text


@dispatch
def count_message_tokens(agent: LlmAgent[AnthropicClientConfig], message: dict):
    enc = tiktoken.get_encoding("cl100k_base")
    if "content" in message:
        # Response from model?
        if isinstance(message["content"], list):
            return sum(count_message_tokens(agent, m) for m in message["content"])
        else:
            return len(enc.encode(message["content"]))
    else:
        # Tool result?
        total = 0
        for k, v in message.items():
            total += len(enc.encode(k)) + len(enc.encode(v))
        return total


@dispatch
def count_message_tokens(agent: LlmAgent[AnthropicClientConfig], message: TextBlock):
    return count_message_tokens(agent, message.citations) + count_message_tokens(
        agent, message.text
    )


@dispatch
def count_message_tokens(agent: LlmAgent[AnthropicClientConfig], message: ToolUseBlock):
    return (
        count_message_tokens(agent, message.id)
        + count_message_tokens(agent, message.input)
        + count_message_tokens(agent, message.name)
    )


@dispatch
def count_message_tokens(agent: LlmAgent[AnthropicClientConfig], message: str):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(message))


@dispatch
def get_summary_text(agent: LlmAgent[AnthropicClientConfig], message: TextBlock):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(message.text))
