# ruff: noqa: F811
import logging
import re
from typing import Any

from lark.exceptions import LarkError
from plum import dispatch

from heracles_evaluation.custom_tool_call_parser import lark_parse_tool
from heracles_evaluation.llm_agent import LlmAgent
from heracles_evaluation.prompt import Prompt
from heracles_evaluation.token_utils import get_token_encoder

logger = logging.getLogger(__name__)


def call_custom_tool_from_string(tools, tool_string):
    try:
        function_call = lark_parse_tool(tool_string)
    except LarkError:
        return "Improperly formatted tool call. Tool call should look like: <tool> a_tool_call(arg1=1, arg2='2') </tool>"

    # I didn't want to deal with escaping logic in Lark, so the parsed string has unparsed escape sequences like \"
    # This should turn \" in to "
    for k, v in function_call.args.items():
        if isinstance(v, str):
            function_call.args[k] = v.encode("utf-8").decode("unicode_escape")

    return tools[function_call.name].function(**function_call.args)


@dispatch
def generate_prompt_for_agent(prompt: Prompt, agent: object):
    raise NotImplementedError(
        f"Cannot generate prompt for client of type {type(agent.client)}."
    )


def extract_tag(tag, string):
    matches = re.findall(rf"<{tag}>([\s\S]*?)<\/{tag}>", string, re.MULTILINE)
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        logger.warning(f"Found multiple {tag} tags in string")
        logger.debug(f"String with multiple tags: {string}")
    return matches[-1]


def extract_answer_tag(string):
    return extract_tag("answer", string)


@dispatch
def is_function_call(agent, message):
    raise NotImplementedError(
        f"is_function_call not implemented for agent type {type(agent)}, message type {type(message)}. Message: {message}"
    )


@dispatch
def get_text_body(message: dict):
    # TODO: it would be better to not need this class and
    # instead wrap Bedrock responses in a type
    if "text" in message:
        return message["text"]
    elif "content" in message:
        return message["content"]
    elif "toolUse" in message:
        t = message["toolUse"]["name"] + "("
        for k, v in message["toolUse"]["input"].items():
            t += f"{k}={v},"
        t += ")"
        return t
    else:
        raise NotImplementedError(f"get_text_body not implemented for dict: {message}")


@dispatch
def get_text_body(message):
    raise NotImplementedError(
        f"get_body not implemented for message type {type(message)}. Message: {message}"
    )


def is_custom_tool_call(agent, message):
    if agent.agent_info.tool_interface != "custom":
        return False
    content = get_text_body(message)
    if content is None:
        return False
    tool_call = extract_tag("tool", content)
    return tool_call is not None


@dispatch
def iterate_messages(agent, messages):
    raise NotImplementedError(
        f"iterate_messages not implemented for agent type {type(agent)}, messages type {type(messages)}"
    )


@dispatch
def call_function(agent, tool_message):
    raise NotImplementedError(
        f"call_function not implemented for agent type {type(agent)}, tool_message type {type(tool_message)}"
    )


@dispatch
def make_tool_response(agent, tool_call_message, result):
    raise NotImplementedError(
        f"make_tool_response not implemented for agent type {type(agent)}, tool_call_message type {type(tool_call_message)}, result type {type(result)}."
    )


# if response is a `list`, then we run this for *any* agent type
@dispatch(precedence=1)
def generate_update_for_history(agent: Any, response: list) -> list:
    return response


@dispatch
def extract_answer(agent, extractor, message):
    raise NotImplementedError(
        f"extract_answer not implemented for agent type {type(agent)}, message type {type(message)}"
    )


@dispatch
def count_message_tokens(agent: LlmAgent, messages: list):
    return sum([count_message_tokens(agent, m) for m in messages])


@dispatch
def count_message_tokens(agent: LlmAgent, message):
    enc = get_token_encoder(agent.model_info.model)
    text = get_text_body(message)
    # Replace endoftext tokens if they exist from Ollama
    text = text.replace("<|endoftext|>", "")
    return len(enc.encode(text))


@dispatch
def count_message_tokens(agent: LlmAgent, message: type(None)):
    return 0


@dispatch
def count_tool_description_tokens(agent: LlmAgent, explicit_tools: dict):
    enc = get_token_encoder(agent.model_info.model)
    logger.debug(
        "Using this string representation for computing tool tokens: ",
        str(explicit_tools),
    )
    return len(enc.encode(str(explicit_tools)))


@dispatch
def count_tool_description_tokens(agent: LlmAgent, explicit_tools: list):
    return sum([count_tool_description_tokens(agent, t) for t in explicit_tools])
