import re
from typing import List, Dict

from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda

from .pipeline import SQLAssistantState
from ..prompts import SQL_GEN_TEMPLATE
from ..utils import get_today_date_en
from ..tools.table.sqlite_database import SQLiteDatabase


_sql_markdown_re = re.compile(r"```sql\s*([\s\S]*?)\s*```", re.DOTALL)
def parse_sql_output(msg_content: str) -> str:
    try:
        match = _sql_markdown_re.search(msg_content)
        if match:
            return match.group(1).strip()
        else:
            raise ValueError("No SQL query found in the content")
    except Exception:
        return msg_content


def preprocess_for_sql_query_generation(
    state: SQLAssistantState,
    database: SQLiteDatabase,
) -> List[AnyMessage]:
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
    system_prompt = SystemMessage(SQL_GEN_TEMPLATE.format(
        table_infos=table_infos,
        date=get_today_date_en(),
        dialect=database.dialect
    ))
    human_message = HumanMessage(content=state["query"])
    return [system_prompt, human_message]


def get_sql_query_generation_chain(chat_model: BaseChatModel) -> Runnable:
    return (
        RunnableLambda(preprocess_for_sql_query_generation)
        | chat_model
        | StrOutputParser()
        | parse_sql_output
    )


async def generate_sql_query(
    state: SQLAssistantState,
    chat_model: BaseChatModel,
) -> SQLAssistantState:
    if not state.get("sql_queries"):
        state["sql_queries"] = []
    sql_gen_chain = get_sql_query_generation_chain(chat_model)
    sql_query = await sql_gen_chain.ainvoke(state)
    state["sql_queries"].append(sql_query)
    return state