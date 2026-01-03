from langchain_community.embeddings import InfinityEmbeddings
from langchain_openai import ChatOpenAI

from pydantic import SecretStr


def get_infinity_embeddings(*, model: str, infinity_api_url: str) -> InfinityEmbeddings:
    """
    Create an Infinity embeddings client.

    Notes:
    - `infinity_api_url` should typically end with `/v1` (e.g. `http://localhost:7797/v1`).
    - The model and url are intentionally NOT hardcoded; pass them from the caller.
    """
    if not infinity_api_url:
        raise ValueError("infinity_api_url is required (e.g. http://localhost:7797/v1)")
    if not model:
        raise ValueError("model is required for InfinityEmbeddings")
    return InfinityEmbeddings(model=model, infinity_api_url=infinity_api_url)


def get_openai_llm_model(model: str, base_url: str, api_key: SecretStr, **kwargs) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        **kwargs
    )
