from typing import Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from .state import SQLAssistantState
from ..prompts import MESSAGE_REWRITING_TEMPLATE
from ..utils import parse_json_output, format_conversation


_message_rewriting_chain_cache: Dict[int, Runnable] = {}
def get_message_rewriting_chain(chat_model: BaseChatModel) -> Runnable:
    # Use model instance ID as cache key (since ChatOpenAI objects aren't hashable)
    chat_model_id = id(chat_model)
    
    if chat_model_id not in _message_rewriting_chain_cache:
        _message_rewriting_chain_cache[chat_model_id] = (
            ChatPromptTemplate([("human", MESSAGE_REWRITING_TEMPLATE)])
            | chat_model
            | StrOutputParser()
            | parse_json_output
        )
    
    return _message_rewriting_chain_cache[chat_model_id]


async def rewrite_message(
    state: SQLAssistantState,
    chat_model: BaseChatModel,
) -> SQLAssistantState:
    conversation = state.get("conversation")
    if not conversation:
        raise ValueError("conversation is required")
    result = await get_message_rewriting_chain(chat_model).ainvoke({
        "formatted_conversation": format_conversation(conversation),
    })
    context = result.get("context", "").strip()
    last_human_message: HumanMessage = HumanMessage(content="")
    for message in conversation[::-1]:
        if message.type == "human":
            last_human_message = message
            break
    state["rewritten_message"] = context + f"\n{last_human_message.content}"
    return state