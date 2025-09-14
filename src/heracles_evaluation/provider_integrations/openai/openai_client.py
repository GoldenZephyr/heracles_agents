from typing import Literal

import openai
from pydantic import BaseModel, Field, PrivateAttr, SecretStr
from pydantic_settings import BaseSettings


class OpenaiClientConfig(BaseSettings):
    client_type: Literal["openai"]
    timeout: int
    auth_key: SecretStr = Field(alias="HERACLES_OPENAI_API_KEY", exclude=True)
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

        if "gpt-5" in model_info.model:
            response = self._client.responses.create(
                model=model_info.model,
                # seed=model_info.seed,
                text=fmt,
                tools=tools,
                input=messages,
                parallel_tool_calls=False,
                reasoning={"effort": "minimal"},
            )
        else:
            response = self._client.responses.create(
                model=model_info.model,
                temperature=model_info.temperature,
                # seed=model_info.seed,
                text=fmt,
                tools=tools,
                input=messages,
                parallel_tool_calls=False,
            )
        return response
