from functools import partial
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AnyMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.language_models import BaseChatModel

from .full_pipeline import SQLAssistantState
from ..utils import get_today_date_vi
from ..tools.table.sqlite_database import SQLiteDatabase
from ..prompts import ANSWER_GEN_TEMPLATE



def preprocess_for_answer_generation(
    state: SQLAssistantState,
    database: SQLiteDatabase,
) -> List[AnyMessage]:
    user_query = state.get("user_query")
    if not user_query:
        raise ValueError("user_query not found in the input")
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
    
    human_message = HumanMessage(content=ANSWER_GEN_TEMPLATE.format(
        date=get_today_date_vi(),
        table_infos=table_infos,
        user_query=user_query,
        sql_query=sql_query,
        db_result=db_result
    ))
    return [human_message]


_answer_generation_chain_cache: Dict[tuple[int, int], Runnable] = {}
def get_answer_generation_chain(chat_model: BaseChatModel, database: SQLiteDatabase) -> Runnable:
    chat_model_id, database_id = id(chat_model), id(database)

    if (chat_model_id, database_id) not in _answer_generation_chain_cache:
        _answer_generation_chain_cache[(chat_model_id, database_id)] = (
            RunnableLambda(partial(preprocess_for_answer_generation, database=database))
            | chat_model
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