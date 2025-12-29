from typing import Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from ..prompts import MESSAGE_REWRITING_TEMPLATE
from .state import SQLAssistantState
from .utils import format_conversation
from ..utils import parse_json_output


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
) -> Dict[str, Dict[str, str]]:
    conversation = state.get("conversation")
    if not conversation:
        raise ValueError("conversation is required")
    rewritten_message = await get_message_rewriting_chain(chat_model).ainvoke({
        "formatted_conversation": format_conversation(conversation)
    })
    context = rewritten_message.get("context", "").strip()
    last_human_message: HumanMessage = HumanMessage(content="")
    for message in conversation[::-1]:
        if message.type == "human":
            last_human_message = message
            break
    rewritten_message = f"{context} {last_human_message.content}"
    state["rewritten_message"] = rewritten_message
    return state