from typing import Literal, Optional

import anthropic
from pydantic import Field, PrivateAttr, SecretStr
from pydantic_settings import BaseSettings


class AnthropicClientConfig(BaseSettings):
    client_type: Literal["anthropic"]
    timeout: int = 120
    auth_key: SecretStr = Field(alias="HERACLES_ANTHROPIC_API_KEY", exclude=True)
    base_url: Optional[str] = None
    default_headers: Optional[dict[str, str]] = None
    _client: object = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        kwargs = {"api_key": self.auth_key.get_secret_value()}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.default_headers:
            kwargs["default_headers"] = self.default_headers
        if self.timeout:
            kwargs["timeout"] = self.timeout
        self._client = anthropic.Anthropic(**kwargs)

    def call(self, model_info, tools, response_format, messages, system=None):
        if response_format != "text":
            raise NotImplementedError(
                "Only `text` format is currently implemented for interfacing with Anthropic"
            )

        kwargs = dict(
            model=model_info.model,
            temperature=model_info.temperature,
            tools=tools,
            messages=messages,
            max_tokens=4096,
        )
        if system:
            kwargs["system"] = system

        messages = self._client.messages.create(**kwargs)
        return messages
