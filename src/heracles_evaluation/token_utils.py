import logging
import tiktoken

logger = logging.getLogger(__name__)


def get_token_encoder(model_name):
    if model_name == "gpt-5":
        model_name = "gpt-5-latest"  # tiktoken is broken
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        logger.warning(
            f"No tiktoken encoder for model: {model_name}. Falling back to cl100k_base"
        )
        enc = tiktoken.get_encoding("cl100k_base")
    return enc
