import logging
import re
from typing import Any, Literal, Optional, Union

from plum import dispatch
from pydantic import BaseModel, Field

import heracles_evaluation.provider_integrations.anthropic.anthropic_agent_integration  # NOQA

# TODO: these should probably get discovered elsewhere
import heracles_evaluation.provider_integrations.openai.openai_agent_integration  # NOQA
from heracles_evaluation.llm_agent import LlmAgent
from heracles_evaluation.prompt import Prompt

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
def is_function_call(agent, message):
    raise NotImplementedError(
        f"is_function_call not implemented for agent type {type(agent)}, message type {type(message)}."
    )


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
        # print("calling with history: ", self.history)
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
