from functools import lru_cache
from langchain_community.embeddings import InfinityEmbeddings
from langchain_openai import ChatOpenAI

from pydantic import SecretStr


@lru_cache()
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


@lru_cache()
def get_openai_llm_model(*, model: str, base_url: str, api_key: SecretStr) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1,
        extra_body = {
            'chat_template_kwargs': {'enable_thinking': False},
            "top_k": 20,
            "mip_p": 0,
        },
    )


@lru_cache()
def get_thinking_openai_llm_model(*, model: str, base_url: str, api_key: SecretStr) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.6,
        top_p=0.95,
        presence_penalty=1,
        extra_body = {
            'chat_template_kwargs': {'enable_thinking': True},
            "top_k": 20,
            "mip_p": 0,
        },
    )
