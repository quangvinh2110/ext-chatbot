from typing import Dict, List
from functools import partial

from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from .state import SQLAssistantState
from ..tools.table.sqlite_database import SQLiteDatabase
from ..prompts import SQL_GEN_TEMPLATE
from ..utils import get_today_date_en, parse_sql_output, format_conversation


def preprocess_for_sql_query_generation(
    state: SQLAssistantState,
    database: SQLiteDatabase,
) -> List[AnyMessage]:
    if state.get("conversation"):
        conversation = state.get("conversation")
    else:
        raise ValueError("conversation is required")
    formatted_conversation = format_conversation(conversation)
    table_infos = "\n\n".join([
        database.get_table_info_no_throw(
            table_name,
            get_col_comments=True,
            sample_count=5,
            column_sample_values=state.get("sample_values", {}).get(table_name),
        )
        for table_name in database.get_usable_table_names()
    ])
    return [HumanMessage(SQL_GEN_TEMPLATE.format(
        date=get_today_date_en(),
        dialect=database.dialect,
        table_infos=table_infos,
        formatted_conversation=formatted_conversation,
    ))]


_sql_query_generation_chain_cache: Dict[tuple[int, int], Runnable] = {}
def get_sql_query_generation_chain(
    chat_model: BaseChatModel, database: SQLiteDatabase
) -> Runnable:
    chat_model_id, database_id = id(chat_model), id(database)
    if (chat_model_id, database_id) not in _sql_query_generation_chain_cache:
        _sql_query_generation_chain_cache[(chat_model_id, database_id)] = (
            RunnableLambda(partial(preprocess_for_sql_query_generation, database=database))
            | chat_model
            | StrOutputParser()
            | parse_sql_output
        )
    
    return _sql_query_generation_chain_cache[(chat_model_id, database_id)]


async def generate_sql_query(
    state: SQLAssistantState,
    chat_model: BaseChatModel,
    database: SQLiteDatabase,
) -> SQLAssistantState:
    if not state.get("sql_queries"):
        state["sql_queries"] = []
    sql_gen_chain = get_sql_query_generation_chain(chat_model, database)
    sql_query = await sql_gen_chain.ainvoke(state)
    state["sql_queries"].append(sql_query)
    return state