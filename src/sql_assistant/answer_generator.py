from typing import List, Dict
from langchain_core.messages import HumanMessage, AnyMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.language_models import BaseChatModel

from .pipeline import SQLAssistantState
from ..utils import get_today_date_vi
from ..tools.table.sqlite_database import SQLiteDatabase
from ..prompts import ANSWER_GEN_TEMPLATE



def preprocess_for_answer_generation(
    state: SQLAssistantState,
    database: SQLiteDatabase,
) -> List[AnyMessage]:
    query = state.get("query")
    if not query:
        raise ValueError("query not found in the input")
    linked_schema: Dict[str, Dict[str, str]] = state.get("linked_schema")
    if not linked_schema:
        raise ValueError("linked_schema not found in the input")
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
    if state.get("db_output_2", {}).get("error", "Error") is None:
        db_result = state.get("db_output_2").get("result")
        sql_query = state.get("sql_query_2")
    elif state.get("db_output_1", {}).get("error", "Error") is None:
        db_result = state.get("db_output_1").get("result")
        sql_query = state.get("sql_query_1")
    else:
        raise ValueError("No valid database result found")
    
    human_message = HumanMessage(content=ANSWER_GEN_TEMPLATE.format(
        date=get_today_date_vi(),
        table_infos=table_infos,
        query=query,
        sql_query=sql_query,
        db_result=db_result
    ))
    return [human_message]


def get_answer_generation_chain(chat_model: BaseChatModel) -> Runnable:
    return (
        RunnableLambda(preprocess_for_answer_generation)
        | chat_model
        | StrOutputParser()
    )


async def generate_answer(
    state: SQLAssistantState,
    chat_model: BaseChatModel,
) -> SQLAssistantState:
    answer_chain = get_answer_generation_chain(chat_model)
    answer = await answer_chain.ainvoke(state)
    state["final_answer"] = answer
    return state