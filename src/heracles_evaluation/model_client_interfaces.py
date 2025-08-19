from typing import Union


from heracles_evaluation.provider_integrations.openai.openai_client import (
    OpenaiClientConfig,
)
from heracles_evaluation.provider_integrations.anthropic.anthropic_client import (
    AnthropicClientConfig,
)

ModelInterfaceConfigType = Union[OpenaiClientConfig, AnthropicClientConfig]


def get_client_union_type():
    """Currently, this function just returns the hard-coded union type of the supported
    integrations, but eventually it could take care of dynamic client plugin registration
    """
    return ModelInterfaceConfigType
