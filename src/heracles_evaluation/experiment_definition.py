from __future__ import annotations

import logging
import os
from typing import Callable

import yaml
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_serializer,
    field_validator,
)

from heracles_evaluation.dsg_interfaces import DsgInterfaceConfigType
from heracles_evaluation.llm_interface import AnalyzedQuestions, EvalQuestion, LlmAgent

logger = logging.getLogger(__name__)


class PipelinePhase(BaseModel):
    name: str
    description: str


class PipelineDescription(BaseModel):
    """Description of an experiment pipeline"""

    name: str
    description: str
    phases: list[PipelinePhase]
    function: Callable[[ExperimentConfiguration], AnalyzedQuestions]

    def get_pipeline_function(self):
        try:
            fn = PipelineRegistry.pipelines[self.name]
        except IndexError as ex:
            print(ex)
            print(
                f"Tool {self.name} not registered in ToolRegistry! Registered tools are {PipelineRegistry.registered_pipeline_summary()}"
            )
        return fn

    def validate_agent_phases(self, experiment_configuration):
        phases_in_pipeline = [p.name for p in self.phases]
        for p in phases_in_pipeline:
            if p not in experiment_configuration.phases:
                phases_in_experiment = list(experiment_configuration.phases.keys())
                raise ValueError(
                    f"Pipeline {self.name} requires agent phase {p}, but {p} is not specified in the experiment configuration. Only {phases_in_experiment} have been specified."
                )


class PipelineRegistry:
    pipelines = {}

    @classmethod
    def registered_pipeline_summary(cls):
        return list(cls.tools.keys())


def register_pipeline(pipeline_description: PipelineDescription):
    name = pipeline_description.name
    if name in PipelineRegistry.pipelines:
        print(f"{name} already has a registered function!")
    else:
        PipelineRegistry.pipelines[name] = pipeline_description


class ExperimentConfiguration(BaseModel):
    # Maps from "type" of model to model interface
    # normally use the "default" type, but in cases
    # where different parts of the pipeline need
    # different model sizes or behaviors, this gives
    # more flexibility
    pipeline: PipelineDescription
    phases: dict[str, LlmAgent]
    dsg_interface: DsgInterfaceConfigType = Field(discriminator="dsg_interface_type")
    questions: list[EvalQuestion]

    @field_validator("questions", mode="before")
    @classmethod
    def load_questions(cls, question_path):
        question_path = os.path.expandvars(question_path)
        logger.debug(f"Loading questions from: {question_path}")
        with open(question_path, "r") as fo:
            yml = yaml.safe_load(fo)

        questions = []
        for q in yml["questions"]:
            try:
                question = EvalQuestion(**q)
            except ValidationError:
                logger.error(
                    "Could not load question: {q}. You probably formatted it incorrectly"
                )
                raise
            questions.append(question)
        logger.debug(f"Loaded {len(questions)} quetions")
        return questions

    @field_validator("pipeline", mode="before")
    @classmethod
    def lookup_pipeline(cls, pipeline_name):
        if pipeline_name in PipelineRegistry.pipelines:
            return PipelineRegistry.pipelines[pipeline_name]
        else:
            raise ValueError(
                f"Requested pipeline name {pipeline_name} not found. Registered pipelines are: {list(PipelineRegistry.pipelines.keys())}"
            )

    @field_serializer("pipeline")
    def serialize_pipeline(self, pipeline: PipelineDescription):
        return pipeline.name

    # TODO: need a custom "after" validator for `pipeline` that ensures the necessary phases have been specified


class ExperimentDescription(BaseModel):
    metadata: dict
    configurations: dict[str, ExperimentConfiguration]


if __name__ == "__main__":
    with open("test.yaml", "r") as fo:
        yml = yaml.safe_load(fo)

    defn = ExperimentConfiguration(**yml)
