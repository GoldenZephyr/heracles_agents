# ruff: noqa: F811
import logging
import time
from typing import Literal, Optional, Union

import boto3
import openai
from openai.types.responses.response_custom_tool_call import (
    ResponseCustomToolCall,
)  # TODO: push this down into the provider integration
from plum import dispatch
from pydantic import BaseModel, Field

from heracles_evaluation.agent_functions import (
    call_function,
    count_message_tokens,
    extract_answer,
    extract_answer_tag,
    generate_prompt_for_agent,
    generate_update_for_history,
    get_text_body,
    is_custom_tool_call,
    is_function_call,
    iterate_messages,
    make_tool_response,
)
from heracles_evaluation.llm_agent import LlmAgent

logger = logging.getLogger(__name__)

BedrockRuntime = boto3.client("bedrock-runtime", "us-east-1")
ThrottlingException = BedrockRuntime.exceptions.ThrottlingException


class SldpComparison(BaseModel):
    comparison_type: Literal["SLDP"]
    relation: str  # equal, subset, superset


class PddlComparison(BaseModel):
    comparison_type: Literal["PDDL"]
    relation: str  # equal, subset, superset


class LLmJudgeComparison(BaseModel):
    comparison_type: Literal["LLM_JUDGE"]


ComparisonType = Union[SldpComparison, PddlComparison, LLmJudgeComparison]


class EvalQuestion(BaseModel):
    name: str
    question: str
    solution: str
    uid: str | int
    correctness_comparator: ComparisonType = Field(discriminator="comparison_type")


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
    input_tokens: int
    output_tokens: int


class AnalyzedQuestion(BaseModel):
    # TODO: do we need to deal with partially-completed response lists?
    question: EvalQuestion
    sequences: list[AgentSequence]
    answer: Optional[str]
    analysis: Optional[QuestionAnalysis]


class AnalyzedQuestions(BaseModel):
    analyzed_questions: list[AnalyzedQuestion]


class AnalyzedExperiment(BaseModel):
    experiment_configurations: dict[str, AnalyzedQuestions]
    metadata: dict = Field(default_factory=dict)


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
        case "ollama":
            explicit_tools = [tool.to_ollama() for tool in agent_info.tools.values()]
        case "bedrock":
            explicit_tools = [tool.to_bedrock() for tool in agent_info.tools.values()]
        case "custom":
            explicit_tools = []
        case "none":
            explicit_tools = []
        case _:
            raise NotImplementedError(
                f"Unknown tool interface: {agent_info.tool_interface}"
            )
    return explicit_tools


def process_answer(agent: LlmAgent, message):
    match agent.agent_info.prompt_settings.output_type:
        case "SLDP_TOOL" | "PDDL_TOOL":
            if isinstance(message, ResponseCustomToolCall):
                # TODO: sanity check that tool matches our requested output tool?
                return message.input

        # case "SLDP" | "PDDL":
        case _:
            answer = extract_answer(agent, extract_answer_tag, message)
            return answer


def is_answer_tool_call(agent: LlmAgent, message):
    # TODO: abstract this check across providers
    # TODO: really we need to explicitly connect the output_type to the tool call name from the LLM.
    # We cannot run both constrained Cypher and constrained SLDP in an agentic framework until we get that right.
    if agent.agent_info.prompt_settings.output_type in [
        "SLDP_TOOL",
        "PDDL_TOOL",
    ] and isinstance(message, ResponseCustomToolCall):
        return True

    return False


def needs_tool_processing(agent: LlmAgent, message):
    if is_answer_tool_call(agent, message):
        return False
    return is_function_call(agent, message) or is_custom_tool_call(agent, message)


@dispatch
def get_summary_text(resp: str):
    return resp


@dispatch
def get_summary_text(resp: type(None)):
    return ""


@dispatch
def get_summary_text(resp: list):
    return "\n".join(get_summary_text(r) for r in resp)


@dispatch
def get_summary_text(resp):
    return get_text_body(resp)


def get_bedrock_block_summary(m):
    if "text" in m:
        return m["text"]
    elif "toolUse" in m:
        summary = m["toolUse"]["name"] + "("
        for k, v in m["toolUse"]["input"].items():
            summary += f"{k}={v},"
        summary += ")"
        return summary
    else:
        raise NotImplementedError(f"Don't know how to summarize: {m}")


@dispatch
def get_summary_text(resp: dict):
    # TODO: Good example of why we need to parse all client responses into types.....
    if "role" in resp and "content" in resp:
        if isinstance(resp["content"], list):
            if "text" in resp["content"][0]:
                # bedrock
                return (
                    resp["role"]
                    + ":"
                    + "\n".join(get_bedrock_block_summary(r) for r in resp["content"])
                )
            else:
                # anthropic
                return (
                    resp["role"]
                    + ":"
                    + "\n".join(get_summary_text(r) for r in resp["content"])
                )
        else:
            return resp["role"] + ": " + resp["content"]
    elif "type" in resp and resp["type"] == "function_call_output" and "output" in resp:
        return "Function result: " + resp["output"]
    elif "toolResult" in resp:
        # bedrock
        return "Tool result: " + "\n".join(
            r["text"] for r in resp["toolResult"]["content"]
        )
    elif "toolUse" in resp:
        # bedrock
        return "Function Call: " + get_bedrock_block_summary(resp)
    else:
        logger.warning(f"Don't know how to turn dict {resp} into elegant string")
        return str(resp)


class AgentContext:
    def __init__(self, agent: LlmAgent):
        self.agent = agent
        self.history = []
        self.n_tool_calls = 0
        self.initial_input_tokens = 0
        self.total_output_tokens = 0

    def initialize_agent(self, prompt):
        self.history = generate_prompt_for_agent(prompt, self.agent)
        self.initial_input_tokens = count_message_tokens(self.agent, self.history)

        logger.info(f"Agent inintialized with: \n{get_summary_text(self.history)}")

    def call_llm(self, history):
        model_info = self.agent.model_info
        logger.debug(f"Calling llm with history: {history}")

        explicit_tools = generate_tools_for_agent(self.agent.agent_info)

        # TODO: what's a reasonable way to set the response format in general?
        # Needs to align with prompt, most likely
        response_format = "text"

        n_ratelimit_retries = 5
        wait_time_s = 60
        for idx in range(n_ratelimit_retries):
            try:
                response = self.agent.client.call(
                    model_info, explicit_tools, response_format, history
                )
            except openai.RateLimitError as ex:
                # TODO: each client should catch their own rate limit errors and then emit a shared single error type that we catch here
                print(ex)
                logging.warning(
                    f"Hit OpenAI rate limit error. Waiting {wait_time_s} seconds. Will retry ({idx} / {n_ratelimit_retries}"
                )
                time.sleep(wait_time_s)
                continue
            except ThrottlingException as ex:
                print(ex)
                logging.warning(
                    f"Hit Bedrock rate limit error. Waiting {wait_time_s} seconds. Will retry ({idx} / {n_ratelimit_retries}"
                )
                time.sleep(wait_time_s)
                continue

            break
        return response

    def handle_response(self, response):
        executed_tool_calls = []
        logger.debug(f"Handling response: {response}")
        for message in iterate_messages(self.agent, response):
            output_tokens = count_message_tokens(self.agent, message)
            self.total_output_tokens += output_tokens
            message_text = get_text_body(message)
            if message_text is not None:
                logger.debug(f"Processing message ({type(message)}: {message_text}")
            if not needs_tool_processing(self.agent, message):
                continue

            self.n_tool_calls += 1

            result = call_function(self.agent, message)

            tool_response = make_tool_response(self.agent, message, result)
            executed_tool_calls.append(tool_response)

        return executed_tool_calls

    def update_history(self, response):
        logger.info(f"History update: \n{get_summary_text(response)}")
        update = generate_update_for_history(self.agent, response)
        self.history += update

    def check_if_done(self, history, response, last_update):
        # If any of the LLM's response messages called an answer tool, we are done
        for message in iterate_messages(self.agent, response):
            if is_answer_tool_call(self.agent, message):
                return True
        # If the LLM didn't call an answer tool, we are done if the LLM didn't call *any* tools.
        return len(last_update) == 0

    def step(self):
        logger.debug("Agent stepping")
        # TODO: Handle timeout and RateLimit errors
        response = self.call_llm(self.history)
        logger.debug(f"Got response: {response}")
        update = self.handle_response(response)
        self.update_history(response)
        self.update_history(update)
        done = self.check_if_done(self.history, response, update)
        return done

    def get_agent_responses(self):
        # TODO: parse the LLM responses into a more useful representation in "parsed_response"
        responses = [
            AgentResponse(
                raw_response=str(resp), parsed_response=get_summary_text(resp)
            )
            for resp in self.history
        ]
        return responses

    def run(self):
        for i in range(self.agent.agent_info.max_iterations):
            done = self.step()
            if done:
                break
        if done:
            answer = process_answer(self.agent, self.history[-1])
        else:
            answer = None
        logger.info(f"Agent exiting. Finished before max iteration cap? {done}")
        logger.info(f"Agent used {self.n_tool_calls} tool calls")
        logger.info(f"Answer: {answer}")
        return done, answer
