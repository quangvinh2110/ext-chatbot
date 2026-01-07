from typing import Dict
import asyncio

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from .state import SQLAssistantState
from ..prompts import MESSAGE_REWRITING_TEMPLATE, TABLE_SELECTING_TEMPLATE
from ..utils import parse_json_output, format_conversation
from ..tools.sqlite_database import SQLiteDatabase


_message_rewriting_chain_cache: Dict[int, Runnable] = {}
def get_message_rewriting_chain(chat_model: BaseChatModel) -> Runnable:
    # Use model instance ID as cache key (since ChatOpenAI objects aren't hashable)
    chat_model_id = id(chat_model)
    
    if chat_model_id not in _message_rewriting_chain_cache:
        _message_rewriting_chain_cache[chat_model_id] = (
                ChatPromptTemplate([("human", MESSAGE_REWRITING_TEMPLATE)])
                | chat_model.bind(temperature=0.0)
                | StrOutputParser()
                | parse_json_output
        )
    
    return _message_rewriting_chain_cache[chat_model_id]


_table_selecting_chain_cache: Dict[int, Runnable] = {}
def get_table_selecting_chain(chat_model: BaseChatModel) -> Runnable:
    # Use model instance ID as cache key (since ChatOpenAI objects aren't hashable)
    chat_model_id = id(chat_model)
    
    if chat_model_id not in _table_selecting_chain_cache:
        _table_selecting_chain_cache[chat_model_id] = (
            ChatPromptTemplate([("human", TABLE_SELECTING_TEMPLATE)])
            | chat_model.bind(temperature=0.0)
            | StrOutputParser()
            | parse_json_output
        )
    
    return _table_selecting_chain_cache[chat_model_id]


async def rewrite_message(
    state: SQLAssistantState,
    chat_model: BaseChatModel,
    database: SQLiteDatabase,
) -> SQLAssistantState:
    conversation = state.get("conversation")
    if not conversation:
        raise ValueError("conversation is required")
    table_overview = database.get_table_overview()
    table_summaries = ""
    if table_overview:
        table_summaries = "\n".join([f"- Table Name: {table['name']}\n  Table Summary: {table['summary']}" for table in table_overview if table["name"] in database.get_usable_table_names()])
    result = await asyncio.gather(
        get_message_rewriting_chain(chat_model).ainvoke({
            "formatted_conversation": format_conversation(conversation),
        }),
        get_table_selecting_chain(chat_model).ainvoke({
            "formatted_conversation": format_conversation(conversation),
            "table_summaries": table_summaries
        })
    )
    relevant_tables = result[1].get("relevant_tables", [])
    is_valid = True
    for table_name in relevant_tables:
        if table_name not in database.get_usable_table_names():
            is_valid = False
            break
    state["relevant_tables"] = relevant_tables if is_valid else database.get_usable_table_names()
    state["context"] = result[0].get("context", "").strip()
    return state