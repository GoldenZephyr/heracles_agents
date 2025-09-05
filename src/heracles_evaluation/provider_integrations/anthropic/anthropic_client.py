from typing import Literal

import anthropic
from pydantic import Field, PrivateAttr, SecretStr
from pydantic_settings import BaseSettings


class AnthropicClientConfig(BaseSettings):
    client_type: Literal["anthropic"]
    auth_key: SecretStr = Field(alias="HERACLES_ANTHROPIC_API_KEY", exclude=True)
    _client: object = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._client = anthropic.Anthropic(api_key=self.auth_key.get_secret_value())

    def call(self, model_info, tools, response_format, messages):
        if response_format != "text":
            raise NotImplementedError(
                "Only `text` format is currently implemented for interfacing with Anthropic"
            )

        messages = self._client.messages.create(
            model=model_info.model,
            temperature=model_info.temperature,
            tools=tools,
            messages=messages,
            max_tokens=4096,
            # system=""
        )
        return messages
