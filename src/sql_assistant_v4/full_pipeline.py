from functools import partial

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from ..tools.sqlite_database import SQLiteDatabase
from .message_rewriter import rewrite_message
from .schema_linker import link_schema
from .sql_generator import generate_sql_query
from .sql_parser import (
    refine_sql_query
)
from .state import SQLAssistantState


async def get_sample_values(
    state: SQLAssistantState,
    database: SQLiteDatabase,
) -> SQLAssistantState:
    full_query: str = ""
    if state.get("context") and isinstance(state.get("context"), str):
        full_query += f"{state.get('context')}"
    if state.get("conversation"):
        last_human_message: HumanMessage = HumanMessage(content="")
        for message in state.get("conversation", [])[::-1]:
            if message.type == "human":
                last_human_message = message
                break
        if isinstance(last_human_message.content, str):
            full_query += f"\n{last_human_message.content}"
    if not full_query:
        state["sample_values"] = {}
        return state
    state["sample_values"] = await database.search_similar_values_from_message(
        full_query,
        k=5
    )
    return state


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


def build_sql_assistant_without_answer_generation(
    chat_model: BaseChatModel,
    database: SQLiteDatabase,
) -> CompiledStateGraph:
    builder = StateGraph(SQLAssistantState)
    # Add nodes
    builder.add_node(
        "rewrite_message",
        partial(
            rewrite_message,
            chat_model=chat_model,
            database=database
        )
    )
    builder.add_node(
        "get_sample_values",
        partial(
            get_sample_values,
            database=database
        )
    )
    builder.add_node(
        "link_schema",
        partial(
            link_schema,
            chat_model=chat_model,
            database=database
        )
    )
    builder.add_node(
        "gen_sql_query",
        partial(
            generate_sql_query, 
            chat_model=chat_model,
            database=database
        )
    )
    builder.add_node(
        "refine_sql_query",
        partial(
            refine_sql_query,
            database=database
        )
    )
    builder.add_node(
        "sql_execution", 
        partial(
            sql_execution,
            database=database
        )
    )

    # Add edges
    builder.add_edge(START, "rewrite_message")
    builder.add_edge("rewrite_message", "get_sample_values")
    builder.add_edge("get_sample_values", "link_schema")
    builder.add_edge("link_schema", "gen_sql_query")
    builder.add_edge("gen_sql_query", "refine_sql_query")
    builder.add_edge("refine_sql_query", "sql_execution")
    builder.add_edge("sql_execution", END)

    return builder.compile()