from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr, PrivateAttr, BaseModel
from typing import Literal, Union
import openai


class OpenaiClientConfig(BaseSettings):
    client_type: Literal["openai"]
    timeout: int
    auth_key: SecretStr = Field(alias="HERACLES_OPENAI_API_KEY")
    _client: object = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._client = openai.OpenAI(
            api_key=self.auth_key.get_secret_value(), timeout=self.timeout
        )

    def call(self, model_info, tools, response_format, messages):
        match response_format:
            case "text":
                # fmt = {"text": {"format": {"type": "text"}}}
                fmt = {"format": {"type": "text"}}
            case "json":
                # fmt = {"text": {"format": {"type": "json_object"}}}
                fmt = {"format": {"type": "json_object"}}
            case BaseModel():
                fmt = response_format
            case _:
                raise ValueError(
                    f"Unknown response_format requested for LLM: {response_format}"
                )

        response = self._client.responses.create(
            model=model_info.model,
            temperature=model_info.temperature,
            # seed=model_info.seed,
            text=fmt,
            tools=tools,
            input=messages,
        )
        return response


class AnthropicClientConfig(BaseSettings):
    client_type: Literal["anthropic"]
    timeout: int
    auth_key: SecretStr = Field(alias="HERACLES_ANTHROPIC_API_KEY")
    _client: object = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._client = None  # TODO: implement


ModelInterfaceConfigType = Union[OpenaiClientConfig, AnthropicClientConfig]
