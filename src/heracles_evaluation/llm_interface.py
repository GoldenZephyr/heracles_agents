from pydantic import BaseModel, Field, field_validator, field_serializer
from typing import Optional, Literal, Union
from heracles_evaluation.model_client_interfaces import ModelInterfaceConfigType
from heracles_evaluation.tool_registry import ToolRegistry
import json
import logging
import re
from openai.types.responses.response_output_message import ResponseOutputMessage
from heracles_evaluation.tool_interface import ToolDescription

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


class ModelInfo(BaseModel):
    """Settings that affect fundamental model performance.

    e.g., model size, temperature, seed.
    Tool calling details are handled elsewhere
    """

    model: str
    temperature: float
    seed: Optional[int] = None


class AgentInfo(BaseModel):
    """Configuration for "agentic" behaviors, e.g., tool calling"""

    tools: list[ToolDescription]
    tool_interface: str  # Openai vs. custom vs. ???
    max_iterations: int

    @field_validator("tools", mode="before")
    @classmethod
    def lookup_tools(cls, tool_names):
        tools = [ToolRegistry.tools[name] for name in tool_names]
        return tools

    @field_serializer("tools")
    def serialize_tools(self, tools):
        return [tool.name for tool in tools]


class LlmAgent(BaseModel):
    agent_info: AgentInfo
    model_info: ModelInfo
    client: ModelInterfaceConfigType = Field(discriminator="client_type")


class AgentContext:
    def __init__(self, agent: LlmAgent):
        self.agent = agent
        self.history = []
        self.n_tool_calls = 0

    def initialize_agent(self, prompt):
        # NOTE: the prompt probably needs to have been constructed with some of
        # the details of the LlmAgent in mind (e.g., tool calling style, max
        # tool calls, etc)
        self.history = prompt.to_openai_json()

    def call_llm(self, history):
        model_info = self.agent.model_info

        # TODO: Notes:
        # 1. this probably isn't where the branching logic on tool type should go?
        # 2. the prompt text may be conditioned on the agent/model info
        if self.agent.agent_info.tool_interface == "openai":
            explicit_tools = [
                tool.to_openai_responses() for tool in self.agent.agent_info.tools
            ]
        else:
            explicit_tools = None

        # TODO: what's a reasonable way to set the response format in general?
        # Needs to align with prompt, most likely
        response_format = "text"
        return self.agent.client.call(
            model_info, explicit_tools, response_format, history
        )

    def handle_response(self, response):
        executed_tool_calls = []
        for message in response.output:
            if message.type != "function_call":
                continue
            self.n_tool_calls += 1
            name = message.name
            args = json.loads(message.arguments)
            result = ToolRegistry.tools[name].function(**args)
            executed_tool_calls.append(
                {
                    "type": "function_call_output",
                    "call_id": message.call_id,
                    "output": str(result),
                }
            )
        return executed_tool_calls

    def update_history(self, update: list):
        self.history += update

    def check_if_done(self, history):
        # If the last message in the history doesn't have tool calls, we are done

        match history[-1]:
            case ResponseOutputMessage():
                return True
            case _:
                return False

    def step(self):
        logger.info("Agent stepping")
        # TODO: Handle timeout and RateLimit errors
        response = self.call_llm(self.history)
        logger.info(f"Got response: {response.output}")
        update = self.handle_response(response)
        self.update_history(response.output)
        self.update_history(update)
        done = self.check_if_done(self.history)
        return done

    def extract_answer(self, message):
        # TODO: eventually this should be generalized to support different answer formats (e.g., structured output?)
        return extract_answer_tag(message.content[0].text)

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
            answer = self.extract_answer(self.history[-1])
        else:
            answer = None
        logger.info(f"Agent exiting. Finished before max iteration cap? {done}")
        logger.info(f"Agent used {self.n_tool_calls} tool calls")
        return done, answer
