from pydantic import Field, BaseModel, field_validator, ValidationError

import yaml
from heracles_evaluation.dsg_interfaces import DsgInterfaceConfigType
from heracles_evaluation.llm_interface import LlmAgent, EvalQuestion

import logging
import os

logger = logging.getLogger(__name__)


# Eventually will be pydantic / pydantic_yaml
class ExperimentDefinition(BaseModel):
    # Maps from "type" of model to model interface
    # normally use the "default" type, but in cases
    # where different parts of the pipeline need
    # different model sizes or behaviors, this gives
    # more flexibility
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

    # task: TaskType


if __name__ == "__main__":
    with open("test.yaml", "r") as fo:
        yml = yaml.safe_load(fo)

    defn = ExperimentDefinition(**yml)
