import re
from typing import Any

from plum import dispatch

from heracles_evaluation.custom_tool_call_parser import lark_parse_tool
from heracles_evaluation.prompt import Prompt


def call_custom_tool_from_string(tools, tool_string):
    function_call = lark_parse_tool(tool_string)
    return tools[function_call.name].function(**function_call.args)


@dispatch
def generate_prompt_for_agent(prompt: Prompt, agent: object):
    raise NotImplementedError(
        f"Cannot generate prompt for client of type {type(agent.client)}."
    )


def extract_tag(tag, string):
    matches = re.findall(f"<{tag}>([\s\S]*?)<\/{tag}>", string, re.MULTILINE)
    if len(matches) > 1:
        # TODO: eventually fail more gracefully?
        raise Exception("Found multiple tags for {tag}!")
    if len(matches) == 0:
        return None
    return matches[0]


def extract_answer_tag(string):
    return extract_tag("answer", string)


@dispatch
def is_function_call(agent, message):
    print("is_function_call called with message: ", message)
    raise NotImplementedError(
        f"is_function_call not implemented for agent type {type(agent)}, message type {type(message)}."
    )


@dispatch
def get_text_body(message):
    raise NotImplementedError(
        f"get_body not implemented for message type {type(message)}."
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
