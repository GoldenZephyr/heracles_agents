from pydantic import BaseModel


class StructuredToolDescription(BaseModel):
    """Description of a structured tool / function"""

    name: str
    description: str
    grammar: str

    def to_openai_responses(self):
        t = {
            "type": "custom",
            "name": self.name,
            "description": self.description,
            "format": {"type": "grammar", "syntax": "lark", "definition": self.grammar},
        }
        return t

    def to_anthropic(self):
        raise NotImplementedError(
            "Structured tool calling not supported for anthropic tools"
        )

    def to_custom(self):
        raise NotImplementedError(
            "Structured tool calling not supported for custom tools"
        )
