# ruff: noqa: F811
import copy
from typing import Callable

from ollama import ChatResponse, Message
from plum import dispatch

from heracles_evaluation.agent_functions import (
    call_custom_tool_from_string,
    extract_tag,
)
from heracles_evaluation.llm_agent import LlmAgent
from heracles_evaluation.prompt import Prompt
from heracles_evaluation.provider_integrations.ollama.ollama_client import (
    OllamaClientConfig,
)
import tiktoken


@dispatch
def generate_prompt_for_agent(prompt: Prompt, agent: LlmAgent[OllamaClientConfig]):
    # explicit_tools = [tool.to_ollama() for tool in agent_info.tools.values()]

    p = copy.deepcopy(prompt)

    if agent.agent_info.tool_interface == "custom":
        # TODO: centralize custom tool prompt logic
        tool_command = "The following tools can be used to help formulate your answer. To call a tool, response with the function name and arguments between a tool tag, like this: <tool> my_function(arg1=1,arg2=2,arg3='3') </tool>.\n"
        for tool in agent.agent_info.tools.values():
            d = tool.to_custom()
            tool_command += d
        p.tool_description = tool_command
    return p.to_anthropic_json()


@dispatch
def iterate_messages(agent: LlmAgent[OllamaClientConfig], response: ChatResponse):
    for m in response.message.tool_calls or []:
        yield m
    yield response.message


@dispatch
def is_function_call(agent: LlmAgent[OllamaClientConfig], tool_call: Message.ToolCall):
    return True


@dispatch
def is_function_call(agent: LlmAgent[OllamaClientConfig], tool_call):
    return False


@dispatch
def call_function(agent: LlmAgent[OllamaClientConfig], tool_call: Message.ToolCall):
    available_tools = agent.agent_info.tools
    name = tool_call.function.name
    return available_tools[name].function(**tool_call.function.arguments)


@dispatch
def call_function(agent: LlmAgent[OllamaClientConfig], tool_call: Message):
    available_tools = agent.agent_info.tools
    tool_string = extract_tag("tool", tool_call.content)
    return call_custom_tool_from_string(available_tools, tool_string)


@dispatch
def make_tool_response(
    agent: LlmAgent[OllamaClientConfig],
    tool_call_message: Message.ToolCall,
    result,
):
    m = {
        "role": "tool",
        "tool_name": tool_call_message.function.name,
        "content": str(result),
    }
    return m


@dispatch
def make_tool_response(agent: LlmAgent[OllamaClientConfig], message: Message, result):
    m = {"role": "user", "content": f"Output of tool call: {result}"}
    return m


@dispatch
def generate_update_for_history(
    agent: LlmAgent[OllamaClientConfig], response: ChatResponse
) -> list:
    return response.message


@dispatch
def extract_answer(
    agent: LlmAgent[OllamaClientConfig],
    extractor: Callable,
    response: ChatResponse,
):
    return extract_answer(agent, extractor, response.message)


@dispatch
def extract_answer(
    agent: LlmAgent[OllamaClientConfig],
    extractor: Callable,
    message: Message,
):
    return extractor(message.content)


@dispatch
def get_text_body(message: Message):
    return message.content


@dispatch
def get_text_body(response: ChatResponse):
    return response.message.content


@dispatch
def get_text_body(tool_call: Message.ToolCall):
    return f"{tool_call.function.name}({tool_call.function.arguments})"


@dispatch
def count_message_tokens(agent: LlmAgent[OllamaClientConfig], message: dict):
    print(message)
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(message["content"]))
