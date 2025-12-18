import json
from pydantic import BaseModel, Field
from functools import partial
from typing import List, Dict, Any

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.language_models import BaseChatModel

from .full_pipeline import SQLAssistantState
from ..utils import get_today_date_vi
from ..tools.table.sqlite_database import SQLiteDatabase
from ..prompts import ANSWER_GEN_TEMPLATE, QUERY_DATABASE_TOOL



class QueryDatabaseInput(BaseModel):
    sql_query: str = Field(description="Câu truy vấn SQLite")


@tool("query_database", args_schema=QueryDatabaseInput)
def query_database(sql_query: str) -> str:
    """Thực hiện câu truy vấn SQLite và trả về kết quả"""
    return ""


def preprocess_for_answer_generation(
    state: SQLAssistantState,
    database: SQLiteDatabase,
) -> List[AnyMessage]:
    conversation = state.get("conversation")
    if not conversation:
        raise ValueError("conversation not found in the input")
    linked_schema: Dict[str, Dict[str, str]] = state.get("linked_schema")
    if not linked_schema:
        raise ValueError("linked_schema not found in the input")
    db_output: Dict[str, Any] = state.get("db_output", {})
    sql_queries: List[str] = state.get("sql_queries", [])
    if not sql_queries:
        raise ValueError("sql_queries not found in the input")
    sql_query = sql_queries[-1]
    if db_output.get("error", "Error") is not None:
        raise ValueError("No valid database result found")
    db_result = db_output.get("result", [])
    
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
    tool_call = {
        "name": "query_database",
        "arguments": {"sql_query": sql_query}
    }
    system_message = SystemMessage(content=ANSWER_GEN_TEMPLATE.format(
        date=get_today_date_vi(),
        table_infos=table_infos,
    ))
    
    sql_conversation = [system_message] + conversation
    sql_conversation.append(AIMessage('<tool_call>\n' + json.dumps(tool_call, ensure_ascii=False) + '\n</tool_call>'))
    sql_conversation.append(HumanMessage(content=str(db_result)))
    return sql_conversation


_answer_generation_chain_cache: Dict[tuple[int, int], Runnable] = {}
def get_answer_generation_chain(chat_model: BaseChatModel, database: SQLiteDatabase) -> Runnable:
    chat_model_id, database_id = id(chat_model), id(database)
    openai_tool_schema = QUERY_DATABASE_TOOL.replace("{{dialect}}", database.dialect)
    if (chat_model_id, database_id) not in _answer_generation_chain_cache:
        _answer_generation_chain_cache[(chat_model_id, database_id)] = (
            RunnableLambda(partial(preprocess_for_answer_generation, database=database))
            | chat_model.bind(tools=[openai_tool_schema])
            | StrOutputParser()
        )
    
    return _answer_generation_chain_cache[(chat_model_id, database_id)]


async def generate_answer(
    state: SQLAssistantState,
    chat_model: BaseChatModel,
    database: SQLiteDatabase,
) -> SQLAssistantState:
    answer_chain = get_answer_generation_chain(chat_model, database)
    answer = await answer_chain.ainvoke(state)
    state["final_answer"] = answer
    return state