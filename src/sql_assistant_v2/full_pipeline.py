from functools import partial

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from langchain_core.language_models import BaseChatModel

from ..tools.table.sqlite_database import SQLiteDatabase
from .schema_linker import link_schema
from .rows_reranker import rerank_rows
from .state import SQLAssistantState
from .message_rewriter import rewrite_message


async def get_sample_values(
    state: SQLAssistantState,
    database: SQLiteDatabase,
) -> SQLAssistantState:
    rewritten_message = state.get("rewritten_message")
    if not rewritten_message:
        state["sample_values"] = {}
        return state
    state["sample_values"] = await database.search_similar_values_from_message(
        rewritten_message,
        k=5
    )
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
        "rerank_rows",
        partial(
            rerank_rows, 
            chat_model=chat_model,
            database=database
        )
    )
    builder.add_edge(START, "rewrite_message")
    builder.add_edge("rewrite_message", "get_sample_values")
    builder.add_edge("get_sample_values", "link_schema")
    builder.add_edge("link_schema", "rerank_rows")
    builder.add_edge("rerank_rows", END)

    return builder.compile()