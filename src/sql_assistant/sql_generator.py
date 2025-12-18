import re
from functools import partial
from typing import List, Dict

from langchain_core.messages import HumanMessage, AnyMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda

from .full_pipeline import SQLAssistantState
from ..prompts import SQL_GEN_TEMPLATE
from ..utils import get_today_date_en
from ..tools.table.sqlite_database import SQLiteDatabase


_sql_markdown_re = re.compile(r"```sql\s*([\s\S]*?)\s*```", re.DOTALL)
def parse_sql_output(msg_content: str) -> str:
    try:
        match = _sql_markdown_re.findall(msg_content)
        if match:
            return match[-1].strip()
        else:
            raise ValueError("No SQL query found in the content")
    except Exception:
        return msg_content


def format_conversation(conversation: List[AnyMessage]) -> str:
    formatted_conversation = ""
    end_index = len(conversation) - 1 
    for ind in range(len(conversation) - 1, -1, -1):
        if conversation[ind].type == "human":
            end_index = ind
            break
    for message in conversation[:end_index]:
        if message.type == "human":
            formatted_conversation += f"Customer: {message.content}\n"
        elif message.type == "ai":
            formatted_conversation += f"Support Team: {message.content}\n"
    
    formatted_conversation += f"\nLatest Customer Message: {conversation[end_index].content}"
    return formatted_conversation


def preprocess_for_sql_query_generation(
    state: SQLAssistantState,
    database: SQLiteDatabase,
) -> List[AnyMessage]:
    linked_schema: Dict[str, Dict[str, str]] = state.get("linked_schema")
    if not linked_schema:
        raise ValueError("linked_schema not found in the input")
    conversation = state.get("conversation")
    if not conversation:
        raise ValueError("conversation not found in the input")
    formatted_conversation = format_conversation(conversation)
    table_infos = "\n\n".join([
        database.get_table_info_no_throw(
            table_name,
            get_col_comments=True,
            allowed_col_names=list(col_types.keys()),
            sample_count=5,
            column_sample_values=state.get("tbl_col_sample_values", {}).get(table_name, None),
        )
        for table_name, col_types in linked_schema.items()
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