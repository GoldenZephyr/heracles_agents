from typing import Literal

from pydantic_settings import BaseSettings

from ollama import chat
from ollama import ChatResponse


class OllamaClientConfig(BaseSettings):
    client_type: Literal["ollama"]

    def __init__(self, **data):
        super().__init__(**data)

    def call(self, model_info, tools, response_format, messages):
        if response_format != "text":
            raise ValueError(
                f"response_format {response_format} not implemented for Ollama!"
            )

        response: ChatResponse = chat(model=model_info.model, messages=messages)

        return response
