import json
import logging
import re
from typing import Literal, Optional, Union, Any

from pydantic import BaseModel, Field, field_serializer, field_validator

from heracles_evaluation.model_client_interfaces import ModelInterfaceConfigType
from heracles_evaluation.prompt import PromptSettings
from heracles_evaluation.tool_interface import ToolDescription
from heracles_evaluation.tool_registry import ToolRegistry
from heracles_evaluation.prompt import Prompt
from heracles_evaluation.model_client_interfaces import (
    OpenaiClientConfig,
    AnthropicClientConfig,
)
from functools import partial
import copy
from anthropic import types as anthropic_types

from plum import dispatch, parametric  # NOQA
from heracles_evaluation.pydantic_discriminated_dispatch import (
    discriminated_union_dispatch,
)

logger = logging.getLogger(__name__)


def extract_answer_tag(string):
    matches = re.findall("<answer>([\s\S]*?)<\/answer>", string, re.MULTILINE)
    if len(matches) > 1:
        # TODO: eventually fail more gracefully?
        raise Exception("Found multiple answeres!")
    if len(matches) == 0:
        return None
    return matches[0]


class SldpComparison(BaseModel):
    comparison_type: Literal["SLDP"]
    relation: str  # equal, subset, superset


class LLmJudgeComparison(BaseModel):
    comparison_type: Literal["LLM_JUDGE"]


ComparisonType = Union[SldpComparison, LLmJudgeComparison]


class EvalQuestion(BaseModel):
    name: str
    question: str
    solution: str
    correctness_comparator: ComparisonType = Field(descriminator="comparison_type")


class AgentResponse(BaseModel):
    # "full" response from the LLM (what format?)
    raw_response: str
    # interpretable response from the LLM (e.g., tool call, parsed final answer)
    parsed_response: Optional[str]


# TODO: seems like this could be useful at some point,
# but currently it's not actually clear that we have
# anything that we want to track per-response.
# Maybe whether a tool call succeeded or something?
# class ResponseAnalysis(BaseModel):
#    # Information that might be relevant for individual LLM responses in the
#    # context of a longer agent interaction.
#
#    # Eventually we might want different types for each "kind" of analysis, but for
#    # now we will accumulate all possible metrics here and default them to false.
#    # It's the job of whatever processes this analysis to decide which of these
#    # flags is meaningful.
#    valid_sldp: bool = False
#    valid_cypher: bool = False

# class AnalyzedResponse(BaseModel):
#    agent_response: AgentResponse
#    response_analysis: ResponseAnalysis


class AgentSequence(BaseModel):
    # What the "purpose" of this agent sequence is. Eventually should be more
    # structured/dispatchable than string?
    description: str

    responses: list[AgentResponse]


class QuestionAnalysis(BaseModel):
    # Information that is relevant about evaluating the response quality of the
    # "whole question"

    valid_answer_format: bool
    correct: bool


class AnalyzedQuestion(BaseModel):
    # TODO: do we need to deal with partially-completed response lists?
    question: EvalQuestion
    sequences: list[AgentSequence]
    analysis: Optional[QuestionAnalysis]


class AnalyzedQuestions(BaseModel):
    analyzed_questions: list[AnalyzedQuestion]


class AnalyzedExperiment(BaseModel):
    experiment_configurations: dict[str, AnalyzedQuestions]


class ModelInfo(BaseModel):
    """Settings that affect fundamental model performance.

    e.g., model size, temperature, seed.
    Tool calling details are handled elsewhere
    """

    model: str
    temperature: float
    seed: Optional[int] = None


def apply_bound_args(tool_name, bound_args):
    args_to_bind = {}
    for arg_name, fields in bound_args.items():
        arg_type = ToolRegistry.get_arg_type(tool_name, arg_name)
        arg_instance = arg_type(**fields)
        args_to_bind[arg_name] = arg_instance
    function = partial(ToolRegistry.tools[tool_name].function, **args_to_bind)
    return function


class AgentInfo(BaseModel):
    """Configuration for "agentic" behaviors, e.g., tool calling"""

    prompt_settings: PromptSettings
    tools: dict[str, ToolDescription]
    tool_interface: str  # Openai vs. custom vs. ???
    max_iterations: int

    @field_validator("tools", mode="before")
    @classmethod
    def lookup_tools(cls, tools):
        tool_descriptions = {}
        for t in tools:
            tool_name = t["name"]
            if tool_name not in ToolRegistry.tools:
                raise ValueError(
                    f"Unknown tool {tool_name}. Known tools: {list(ToolRegistry.tools.keys())}"
                )
            if "bound_args" in t:
                function = apply_bound_args(tool_name, t["bound_args"])
                resolved_tool = copy.deepcopy(ToolRegistry.tools[tool_name])
                resolved_tool.function = function
            else:
                resolved_tool = ToolRegistry.tools[tool_name]
            tool_descriptions[tool_name] = resolved_tool

        return tool_descriptions

    @field_serializer("tools")
    def serialize_tools(self, tools):
        return [tool.name for tool in tools]


@parametric
@discriminated_union_dispatch("client")
class LlmAgent[T](BaseModel):
    agent_info: AgentInfo
    model_info: ModelInfo
    client: ModelInterfaceConfigType = Field(discriminator="client_type")


@dispatch
def generate_prompt_for_agent(prompt: Prompt, agent: LlmAgent[AnthropicClientConfig]):
    return prompt.to_anthropic_json()


@dispatch
def generate_prompt_for_agent(prompt: Prompt, agent: LlmAgent[OpenaiClientConfig]):
    return prompt.to_openai_json()


@dispatch
def generate_prompt_for_agent(prompt: Prompt, agent: object):
    raise NotImplementedError(
        f"Cannot generate prompt for client of type {type(agent.client)}."
    )


def generate_tools_for_agent(agent_info):
    # TODO: if we want to go all the way with dynamic dispatch, the tool rendering functions
    # also need to be made dynamic dispatch
    match agent_info.tool_interface:
        case "openai":
            explicit_tools = [
                tool.to_openai_responses() for tool in agent_info.tools.values()
            ]
        case "anthropic":
            explicit_tools = [tool.to_anthropic() for tool in agent_info.tools.values()]
        case "custom":
            explicit_tools = [tool.to_custom() for tool in agent_info.tools.values()]
        case _:
            raise NotImplementedError(
                f"Unknown tool interface: {agent_info.tool_interface}"
            )
    return explicit_tools


@dispatch
def is_function_call(agent: LlmAgent[AnthropicClientConfig], message):
    return message.type == "tool_use"


@dispatch
def is_function_call(agent: LlmAgent[OpenaiClientConfig], message):
    return message.type == "function_call"


@dispatch
def iterate_messages(agent: LlmAgent[AnthropicClientConfig], messages):
    for m in messages.content:
        yield m


@dispatch
def iterate_messages(agent: LlmAgent[OpenaiClientConfig], messages):
    for m in messages.output:
        yield m


@dispatch
def call_function(agent: LlmAgent[AnthropicClientConfig], tool_message):
    available_tools = agent.agent_info.tools
    name = tool_message.name
    args = tool_message.input
    # TODO: verify legal tool name
    return available_tools[name].function(**args)


@dispatch
def call_function(agent: LlmAgent[OpenaiClientConfig], tool_message):
    available_tools = agent.agent_info.tools
    name = tool_message.name
    args = json.loads(tool_message.arguments)
    # TODO: verify legal tool name
    return available_tools[name].function(**args)


@dispatch
def make_tool_response(
    agent: LlmAgent[AnthropicClientConfig], tool_call_message, result
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
def make_tool_response(agent: LlmAgent[OpenaiClientConfig], tool_call_message, result):
    m = {
        "type": "function_call_output",
        "call_id": tool_call_message.call_id,
        "output": str(result),
    }
    return m


@dispatch
def generate_update_for_history(
    agent: LlmAgent[AnthropicClientConfig], response
) -> list:
    m = anthropic_types.MessageParam(role="assistant", content=response.content)
    return [m]


@dispatch
def generate_update_for_history(agent: LlmAgent[OpenaiClientConfig], response) -> list:
    return response.output


@dispatch(precedence=1)
def generate_update_for_history(agent: Any, response: list) -> list:
    return response


@dispatch
def extract_answer(agent: LlmAgent[AnthropicClientConfig], extractor, message):
    return extractor(message["content"][0].text)


@dispatch
def extract_answer(agent: LlmAgent[OpenaiClientConfig], extractor, message):
    return extractor(message.content[0].text)


class AgentContext:
    def __init__(self, agent: LlmAgent):
        self.agent = agent
        self.history = []
        self.n_tool_calls = 0

    def initialize_agent(self, prompt):
        self.history = generate_prompt_for_agent(prompt, self.agent)

    def call_llm(self, history):
        model_info = self.agent.model_info

        explicit_tools = generate_tools_for_agent(self.agent.agent_info)

        # TODO: what's a reasonable way to set the response format in general?
        # Needs to align with prompt, most likely
        response_format = "text"
        return self.agent.client.call(
            model_info, explicit_tools, response_format, history
        )

    def handle_response(self, response):
        executed_tool_calls = []
        for message in iterate_messages(self.agent, response):
            if not is_function_call(self.agent, message):
                continue
            self.n_tool_calls += 1

            result = call_function(self.agent, message)

            tool_response = make_tool_response(self.agent, message, result)
            executed_tool_calls.append(tool_response)

        return executed_tool_calls

    def update_history(self, response):
        update = generate_update_for_history(self.agent, response)
        self.history += update

    # def check_if_done(self, history):
    #    # If the last message in the history doesn't have tool calls, we are done
    #    return self.done

    def step(self):
        logger.info("Agent stepping")
        # TODO: Handle timeout and RateLimit errors
        print("calling with history: ", self.history)
        response = self.call_llm(self.history)
        logger.info(f"Got response: {response}")
        update = self.handle_response(response)
        self.update_history(response)
        self.update_history(update)
        done = len(update) == 0
        # done = self.check_if_done(self.history)
        return done

    def get_agent_responses(self):
        # TODO: parse the LLM responses into a more useful representation in "parsed_response"
        responses = [
            AgentResponse(raw_response=str(resp), parsed_response="TODO")
            for resp in self.history
        ]
        return responses

    def run(self):
        for i in range(self.agent.agent_info.max_iterations):
            done = self.step()
            if done:
                break
        if done:
            # TODO: eventually this should be generalized to support different answer formats (e.g., structured output?)
            # Also need to handle answers that are supplied via function call (e.g., for OpenAI constrained decoding outputs)
            answer = extract_answer(self.agent, extract_answer_tag, self.history[-1])
        else:
            answer = None
        logger.info(f"Agent exiting. Finished before max iteration cap? {done}")
        logger.info(f"Agent used {self.n_tool_calls} tool calls")
        return done, answer
