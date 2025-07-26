from pydantic import BaseModel, Field, field_validator, field_serializer
from typing import Optional, Literal, Union, Any, Callable
from model_client_interfaces import ModelInterfaceConfigType
from heracles_evaluation.tool_registry import ToolRegistry
import json
from dataclasses import dataclass
import logging
import re
from openai.types.responses.response_output_message import ResponseOutputMessage

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


class ModelInfo(BaseModel):
    """Settings that affect fundamental model performance.

    e.g., model size, temperature, seed.
    Tool calling details are handled elsewhere
    """

    model: str
    temperature: float
    seed: Optional[int] = None


def type_to_string(typ):
    match typ():
        case str():
            return "string"
        case float():
            return "float"
        case int():
            return "int"
        case dict():
            return "dict"
        case set():
            return "set"
        case list():
            return "list"


@dataclass
class FunctionParameter:
    """Description of a single parameter for a tool/function call"""

    name: str
    param_type: type
    param_description: str
    required: bool = True
    enum_values: Optional[Any] = None

    def to_openai_responses(self):
        d = {
            self.name: {
                "type": type_to_string(self.param_type),
                "description": self.param_description,
            }
        }
        if self.enum_values is not None:
            d[self.name]["enum"] = self.enum_values
        return d


@dataclass
class ToolDescription:
    """Description of a tool / function"""

    name: str
    description: str
    parameters: list[FunctionParameter]
    function: Callable

    def get_tool_function(self):
        try:
            fn = ToolRegistry.lookup(self.name)
        except IndexError as ex:
            print(ex)
            print(
                f"Tool {self.name} not registered in ToolRegistry! Registered tools are {ToolRegistry.registered_tool_summary()}"
            )
        return fn

    def to_openai_responses(self):
        parameter_properties = {}
        for p in self.parameters:
            parameter_properties |= p.to_openai_responses()

        required = [p.name for p in self.parameters if p.required]

        parameter_descriptions = {
            "type": "object",
            "properties": parameter_properties,
            "required": required,
            "additionalProperties": False,
        }

        t = {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": parameter_descriptions,
        }
        print("tool formatted: ")
        print(t)
        return t


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
        return self.agent.client.call(model_info, explicit_tools, "text", history)

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
