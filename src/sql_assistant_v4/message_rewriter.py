from typing import Dict, List
import asyncio

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel

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
    tasks = []
    if len(conversation) >= 2:
        tasks.append(get_message_rewriting_chain(chat_model).ainvoke({
            "formatted_conversation": format_conversation(conversation),
        }))
    else:
        async def default_message_rewriting_chain():
            return {"context": ""}
        tasks.append(default_message_rewriting_chain())
    usable_table_names = database.get_usable_table_names()
    num_columns_list = []
    for table_name in usable_table_names:
        column_names = database.get_column_names(table_name)
        if isinstance(column_names, list):
            num_columns_list.append(len(column_names))
    if len(usable_table_names) >= 2 and sum(num_columns_list) >= 15:
        tasks.append(get_table_selecting_chain(chat_model).ainvoke({
            "formatted_conversation": format_conversation(conversation),
            "table_summaries": table_summaries
        }))
    else:
        async def default_table_selecting_chain():
            return {"relevant_tables": database.get_usable_table_names()}
        tasks.append(default_table_selecting_chain())
    result = await asyncio.gather(*tasks)
    relevant_tables = result[1].get("relevant_tables", [])
    valid_relevant_tables: List[str] = []
    for table_name in relevant_tables:
        if table_name in database.get_usable_table_names():
            valid_relevant_tables.append(table_name)
    state["relevant_tables"] = valid_relevant_tables
    state["context"] = result[0].get("context", "").strip()
    return state