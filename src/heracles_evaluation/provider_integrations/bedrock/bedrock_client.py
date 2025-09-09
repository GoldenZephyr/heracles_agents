from typing import Literal

import boto3
from pydantic import PrivateAttr
from pydantic_settings import BaseSettings

model_name_to_bedrock_model_id = {
    "bedrock_claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "bedrock_claude-4-sonnet": "us.anthropic.claude-sonnet-4-20250514-v1:0",
}


class BedrockClientConfig(BaseSettings):
    client_type: Literal["bedrock"]
    timeout: int
    _client: object = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._client = boto3.client("bedrock-runtime", region_name="us-east-1")

    def call(self, model_info, tools, response_format, messages):
        # TODO: tools
        model_id = model_name_to_bedrock_model_id[model_info.model]
        req = {"modelId": model_id, "messages": messages}
        # if system_prompt:
        #    req["system"] = [{"text": system_prompt}]
        req["inferenceConfig"] = {}
        req["inferenceConfig"] = {"temperature": 0.7}

        response = self._client.converse(**req)
        return response
