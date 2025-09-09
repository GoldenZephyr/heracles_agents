import logging
import os
from typing import List, Optional

import yaml
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)


class InContextExample(BaseModel):
    user: str
    assistant: str
    system: Optional[str] = None

    def to_openai_json(self):
        parts = []

        if self.system:
            system_part = {"role": "developer", "content": self.system}
            parts.append(system_part)

        user_part = {"role": "user", "content": self.user}
        parts.append(user_part)
        assistant_part = {"role": "assistant", "content": self.assistant}
        parts.append(assistant_part)

        return parts


class Prompt(BaseModel):
    system: str
    tool_description: Optional[str] = None
    in_context_examples_preamble: Optional[str] = None
    in_context_examples: Optional[List[InContextExample]] = None
    novel_instruction_preamble: Optional[str] = None
    novel_instruction: Optional[str] = None
    novel_instruction_template: Optional[str] = None
    answer_semantic_guidance: Optional[str] = None
    answer_formatting_guidance: Optional[str] = None

    def to_openai_json(self, novel_instruction=None):
        # NOTE: Currently for openai we set a bunch of things as `developer`.
        # Anthropic doesn't quite have the same notion of developer.
        # They have a system prompt, but it's set in a different places from
        # the normal messages, and seems meant for pretty short descriptions
        # of what the model should be doing.
        # For consistency, we may want to turn these `developer` roles into
        # `user` ? This would at least make it more consistent with the Anthropic interface.
        # I think we shouldn't worry too much about this until after we implement a third provider.
        if self.novel_instruction is None and novel_instruction is None:
            raise ValueError(
                "novel_instruction must be set either at Prompt initialization or as an argument to `to_openai_json`"
            )
        prompt = [{"role": "developer", "content": self.system}]

        if self.tool_description:
            prompt.append({"role": "developer", "content": self.tool_description})

        if self.in_context_examples_preamble:
            prompt.append(
                {
                    "role": "developer",
                    "content": self.in_context_examples_preamble,
                }
            )

        if self.in_context_examples:
            for e in self.in_context_examples:
                prompt += e.to_openai_json()

        if self.novel_instruction_preamble:
            prompt.append(
                {"role": "developer", "content": self.novel_instruction_preamble}
            )

        if novel_instruction:
            if self.novel_instruction:
                logger.warning(
                    f"Overriding default novel instruction `{self.novel_instruction}` with new instruction `{novel_instruction}`"
                )
            prompt.append({"role": "user", "content": novel_instruction})
        elif self.novel_instruction:
            prompt.append({"role": "user", "content": self.novel_instruction})

        if self.answer_semantic_guidance:
            prompt.append(
                {"role": "developer", "content": self.answer_semantic_guidance}
            )

        if self.answer_formatting_guidance:
            prompt.append(
                {"role": "developer", "content": self.answer_formatting_guidance}
            )

        return prompt

    def to_anthropic_json(self, novel_instruction=None):
        if self.novel_instruction is None and novel_instruction is None:
            raise ValueError(
                "novel_instruction must be set either at Prompt initialization or as an argument to `to_anthropic_json`"
            )
        prompt = [{"role": "user", "content": self.system}]

        if self.tool_description:
            prompt.append({"role": "user", "content": self.tool_description})

        if self.in_context_examples_preamble:
            prompt.append(
                {
                    "role": "user",
                    "content": self.in_context_examples_preamble,
                }
            )

        if self.in_context_examples:
            for e in self.in_context_examples:
                prompt += e.to_openai_json()

        if self.novel_instruction_preamble:
            prompt.append({"role": "user", "content": self.novel_instruction_preamble})

        if novel_instruction:
            if self.novel_instruction:
                logger.warning(
                    f"Overriding default novel instruction `{self.novel_instruction}` with new instruction `{novel_instruction}`"
                )
            prompt.append({"role": "user", "content": novel_instruction})
        elif self.novel_instruction:
            prompt.append({"role": "user", "content": self.novel_instruction})

        if self.answer_semantic_guidance:
            prompt.append({"role": "user", "content": self.answer_semantic_guidance})

        if self.answer_formatting_guidance:
            prompt.append({"role": "user", "content": self.answer_formatting_guidance})

        return prompt

    def to_bedrock_json(self, novel_instruction=None):
        if self.novel_instruction is None and novel_instruction is None:
            raise ValueError(
                "novel_instruction must be set either at Prompt initialization or as an argument to `to_anthropic_json`"
            )
        prompt = [{"role": "user", "content": [{"text": self.system}]}]

        if self.tool_description:
            prompt.append(
                {"role": "user", "content": [{"text": self.tool_description}]}
            )

        if self.in_context_examples_preamble:
            prompt.append(
                {
                    "role": "user",
                    "content": [{"text": self.in_context_examples_preamble}],
                }
            )

        if self.in_context_examples:
            for e in self.in_context_examples:
                prompt += e.to_openai_json()

        if self.novel_instruction_preamble:
            prompt.append(
                {"role": "user", "content": [{"text": self.novel_instruction_preamble}]}
            )

        if novel_instruction:
            if self.novel_instruction:
                logger.warning(
                    f"Overriding default novel instruction `{self.novel_instruction}` with new instruction `{novel_instruction}`"
                )
            prompt.append({"role": "user", "content": [{"text": novel_instruction}]})
        elif self.novel_instruction:
            prompt.append(
                {"role": "user", "content": [{"text": self.novel_instruction}]}
            )

        if self.answer_semantic_guidance:
            prompt.append(
                {"role": "user", "content": [{"text": self.answer_semantic_guidance}]}
            )

        if self.answer_formatting_guidance:
            prompt.append(
                {"role": "user", "content": [{"text": self.answer_formatting_guidance}]}
            )

        return prompt

    def __repr__(self):
        return repr(self.to_openai_json("<Question>"))


class PromptSettings(BaseModel):
    base_prompt: Prompt
    output_type: Optional[str] = None
    sldp_answer_type_hint: bool = False  # TODO: move this....

    @field_validator("base_prompt", mode="before")
    @classmethod
    def load_prompt(cls, prompt_path):
        # TODO: Ideally we would handle the case where a full Prompt is
        # specified in the input yaml file (and not just a path), to support
        # round-tripping where we want to dump the experiment back out to the
        # final (in which case we will have lost track of the original file
        # path. Instead we would just dump the full prompt structure
        match prompt_path:
            case str():
                prompt_path = os.path.expandvars(prompt_path)
                if not os.path.exists(prompt_path):
                    raise ValueError(f"Prompt path does not exist: {prompt_path}")
                with open(prompt_path, "r") as fo:
                    prompt_yaml = yaml.safe_load(fo)
                return Prompt(**prompt_yaml)
            case dict():
                return prompt_path
            case Prompt():
                return prompt_path
            case _:
                raise ValueError(
                    f"PromptSettings cannot initialize base_prompt from type {type(prompt_path)}"
                )


def get_sldp_format_description():
    return """
Please format your response according to the SLDP Equality Language:

## SLDP Equality Language

To evaluate if an answer is correct, we need to define a sense of equality.
This is rather tricky, because there are different senses in which things can
be equal.

We need to handle Lists, Sets, Dictionaries, and Points.  Lists are equal if
each element is equal, sets A and B are equal if A ⊆ B and B ⊆ A, Dictionaries
are equal if the sets of their keys are equal and the value for each key
matches between dictionaries, and two points are equal if they are  within some
tolerance. Of course primitive numbers and strings can also be compared for
equality. We support arbitrary compositions of these containers.

### Syntax

A primitive string is a sequence of alphanumeric characters (with no quotation).

A primitive number is a floating point representation of a number.

A `list` is written as `[element1, element2, ... elementN]`

A `set` is written as `<element1, element2, ... elementN>`

A `dict` is written as `{k1: v1, k2: v2}`

A `point` is written as `POINT(x y z)` (note the lack of comma)
"""


def get_sldp_answer_tag_text():
    return """
### Denoting Final Answer:

Format your final answer (*not* any intermediate tool calls) as an SLDP
expression wrapped between <answer> and </answer> tags, such as
<answer><1,2,3></answer>. Only a single pair of answer tags should appear in
your solution.
"""
