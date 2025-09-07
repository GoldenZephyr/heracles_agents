from typing import Literal

from ollama import ChatResponse, chat
from pydantic import PrivateAttr
from pydantic_settings import BaseSettings


class OllamaClientConfig(BaseSettings):
    client_type: Literal["ollama"]
    _chat_func: object = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        if self._chat_func is None:
            self._chat_func = chat

    def call(self, model_info, tools, response_format, messages):
        if response_format != "text":
            raise ValueError(
                f"response_format {response_format} not implemented for Ollama!"
            )

        response: ChatResponse = self._chat_func(
            model=model_info.model, messages=messages, tools=tools
        )

        return response
