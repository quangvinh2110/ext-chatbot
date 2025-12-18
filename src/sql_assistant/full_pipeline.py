from typing import TypedDict, Dict, List, Any, Literal
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
    user_query: str
    linked_schema: Dict[str, Dict[str, str]]
    sql_queries: List[str]
    predicate_values: List[Dict[str, Any]]
    tbl_col_sample_values: Dict[str, Dict[str, List[Any]]]
    db_output: Dict[str, Any]
    final_answer: str


def retry_condition(
    state: SQLAssistantState
) -> Literal["gen_sql_query_2", "restrict_select_columns"]:
    predicate_values = state.get("predicate_values")
    if not predicate_values:
        return "restrict_select_columns"
    similar_predicate_values = state.get("tbl_col_sample_values")
    if not similar_predicate_values:
        return "restrict_select_columns"
    
    # Check if all original predicate values are found in the similar values
    all_found = True
    for pred_value in predicate_values:
        table_name = pred_value["table_name"]
        column_name = pred_value["column_name"]
        original_value = pred_value["value"]
        
        # Get the list of similar values for this table/column pair
        similar_values = similar_predicate_values.get(table_name, {}).get(column_name, [])
        
        # If the original value is NOT found in similar values, we need to rewrite
        if original_value not in similar_values:
            all_found = False
            break
    
    # If all original values were found in similar values, we don't need to rewrite
    if all_found:
        return "restrict_select_columns"
    
    # If any original value was not found in similar values, we should rewrite
    return "gen_sql_query_2"
    


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
            chat_model=chat_model,
            database=database
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
            chat_model=chat_model,
            database=database
        )
    )
    builder.add_node(
        "restrict_select_columns",
        partial(
            restrict_select_columns,
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
    builder.add_node(
        "answer_generation", 
        partial(
            generate_answer,
            chat_model=chat_model,
            database=database
        )
    )

    # Add edges
    builder.add_edge(START, "link_schema")
    builder.add_edge("link_schema", "gen_sql_query_1")
    builder.add_edge("gen_sql_query_1", "get_predicate_values")
    builder.add_edge("get_predicate_values", "get_similar_predicate_values")
    builder.add_conditional_edges(
        "get_similar_predicate_values",
        retry_condition,
    )
    builder.add_edge("gen_sql_query_2", "restrict_select_columns")
    builder.add_edge("restrict_select_columns", "sql_execution")
    builder.add_edge("sql_execution", "answer_generation")
    builder.add_edge("answer_generation", END)

    return builder.compile()