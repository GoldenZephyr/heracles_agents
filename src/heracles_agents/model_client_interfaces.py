from typing import Union

from heracles_agents.provider_integrations.anthropic.anthropic_client import (
    AnthropicClientConfig,
)
from heracles_agents.provider_integrations.bedrock.bedrock_client import (
    BedrockClientConfig,
)
from heracles_agents.provider_integrations.ollama.ollama_client import (
    OllamaClientConfig,
)
from heracles_agents.provider_integrations.openai.openai_client import (
    OpenaiClientConfig,
)
from heracles_agents.provider_integrations.openai.openai_completions_client import (
    OpenaiCompletionsClientConfig,
)

ModelInterfaceConfigType = Union[
    OpenaiClientConfig,
    OpenaiCompletionsClientConfig,
    AnthropicClientConfig,
    OllamaClientConfig,
    BedrockClientConfig,
]


def get_client_union_type():
    """Currently, this function just returns the hard-coded union type of the supported
    integrations, but eventually it could take care of dynamic client plugin registration
    """
    return ModelInterfaceConfigType
