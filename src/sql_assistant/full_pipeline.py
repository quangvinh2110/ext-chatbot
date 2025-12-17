from typing import TypedDict, Dict, List, Any
from functools import partial

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from langchain_core.language_models import BaseChatModel
from ..tools.table.sqlite_database import SQLiteDatabase
from .schema_linker import link_schema
from .sql_generator import generate_sql_query
from .sql_parser import get_predicate_values, get_similar_predicate_values
from .answer_generator import generate_answer


class SQLAssistantState(TypedDict):
    query: str
    linked_schema: Dict[str, Dict[str, str]]
    sql_queries: List[str]
    predicate_values: List[Dict[str, Any]]
    tbl_col_sample_values: Dict[str, Dict[str, List[Any]]]
    db_output: Dict[str, Any]
    final_answer: str


async def sql_execution(
    state: SQLAssistantState,
    database: SQLiteDatabase,
) -> SQLAssistantState:
    sql_queries = state.get("sql_queries", [])
    if not sql_queries:
        raise ValueError("SQL queries are required")
    sql_query = sql_queries[-1]
    state["db_output"] = await database.run_no_throw(sql_query, include_columns=True)
    return state


def build_sql_assistant_pipeline(
    chat_model: BaseChatModel,
    database: SQLiteDatabase,
) -> CompiledStateGraph:
    builder = StateGraph(SQLAssistantState)
    # Add nodes
    builder.add_node(
        "link_schema", 
        partial(
            link_schema,
            chat_model=chat_model,
            database=database
        )
    )
    builder.add_node(
        "gen_sql_query_1",
        partial(
            generate_sql_query, 
            chat_model=chat_model
        )
    )
    builder.add_node(
        "get_predicate_values", 
        partial(
            get_predicate_values, 
            database=database
        )
    )
    builder.add_node(
        "get_similar_predicate_values", 
        partial(
            get_similar_predicate_values, 
            database=database
        )
    )
    builder.add_node(
        "gen_sql_query_2",
        partial(
            generate_sql_query,
            chat_model=chat_model
        )
    )
    builder.add_node(
        "sql_execution", 
        partial(
            sql_execution,
            database=database
        )
    )
    builder.add_node(
        "answer_generation", 
        partial(
            generate_answer,
            chat_model=chat_model
        )
    )

    # Add edges
    builder.add_edge(START, "link_schema")
    builder.add_edge("link_schema", "gen_sql_query_1")
    builder.add_edge("gen_sql_query_1", "get_predicate_values")
    builder.add_edge("get_predicate_values", "get_similar_predicate_values")
    builder.add_edge("get_similar_predicate_values", "gen_sql_query_2")
    builder.add_edge("gen_sql_query_2", "sql_execution")
    builder.add_edge("sql_execution", "answer_generation")
    builder.add_edge("answer_generation", END)

    return builder.compile()