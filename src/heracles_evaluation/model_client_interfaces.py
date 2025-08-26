from typing import Union

from heracles_evaluation.provider_integrations.anthropic.anthropic_client import (
    AnthropicClientConfig,
)
from heracles_evaluation.provider_integrations.ollama.ollama_client import (
    OllamaClientConfig,
)
from heracles_evaluation.provider_integrations.openai.openai_client import (
    OpenaiClientConfig,
)

ModelInterfaceConfigType = Union[
    OpenaiClientConfig, AnthropicClientConfig, OllamaClientConfig
]


def get_client_union_type():
    """Currently, this function just returns the hard-coded union type of the supported
    integrations, but eventually it could take care of dynamic client plugin registration
    """
    return ModelInterfaceConfigType
