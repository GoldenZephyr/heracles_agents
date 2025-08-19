from plum import dispatch
from heracles_evaluation.llm_agent import LlmAgent
from heracles_evaluation.provider_integrations.anthropic.anthropic_client import (
    AnthropicClientConfig,
)
from heracles_evaluation.prompt import Prompt
from anthropic import types as anthropic_types
from anthropic.types.tool_use_block import ToolUseBlock
from anthropic.types.message import Message


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
def generate_update_for_history(
    agent: LlmAgent[AnthropicClientConfig], response: Message
) -> list:
    m = anthropic_types.MessageParam(role="assistant", content=response.content)
    return [m]


@dispatch
def extract_answer(agent: LlmAgent[AnthropicClientConfig], extractor, message: dict):
    return extractor(message["content"][0].text)
