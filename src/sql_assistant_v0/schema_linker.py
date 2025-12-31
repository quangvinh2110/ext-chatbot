import asyncio
from typing import Optional, List, Dict, Any

from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .state import SQLAssistantState
from ..tools.table.sqlite_database import SQLiteDatabase
from ..prompts import SCHEMA_LINKING_TEMPLATE


# Cache for schema linking chains keyed by model instance ID
_schema_linking_chain_cache: Dict[int, Runnable] = {}
def get_schema_linking_chain(chat_model: BaseChatModel) -> Runnable:
    # Use model instance ID as cache key (since ChatOpenAI objects aren't hashable)
    chat_model_id = id(chat_model)
    
    if chat_model_id not in _schema_linking_chain_cache:
        _schema_linking_chain_cache[chat_model_id] = (
            ChatPromptTemplate([("human", SCHEMA_LINKING_TEMPLATE)])
            | chat_model
            | JsonOutputParser()
        )
    
    return _schema_linking_chain_cache[chat_model_id]


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
    if not formatted_conversation:
        formatted_conversation = "No conversation history\n"
    formatted_conversation += f"\nLatest Customer Message: {conversation[end_index].content}"
    return formatted_conversation


async def _link_schema_one(
    conversation: List[AnyMessage],
    table_name: str,
    chat_model: BaseChatModel,
    database: SQLiteDatabase,
    allowed_col_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    try:
        column_names = database.get_column_names(table_name)
        if isinstance(column_names, list) and len(column_names) <= 5:
            return {
                "input_item": {
                    "table_name": table_name,
                    "conversation": conversation,
                    "allowed_col_names": allowed_col_names
                },
                "filtered_schema": (table_name, column_names),
                "error": None
            }
        table_overview = database.get_table_overview()
        table_summary = ""
        for table in table_overview:
            if table["name"] == table_name:
                table_summary = table.get("summary", "")
                break
        table_info = database.get_table_info_no_throw(
            table_name,
            get_col_comments=True,
            allowed_col_names=allowed_col_names,
            sample_count=3
        )
        result = await get_schema_linking_chain(chat_model).ainvoke({
            "table_summary": table_summary,
            "table_info": table_info, 
            "formatted_conversation": format_conversation(conversation), 
            "dialect": database.dialect
        })
        
        if "is_related" not in result or result["is_related"] not in ["Y", "N"]:
            raise ValueError("Invalid response from schema linking chain")
        if result["is_related"] == "Y" and not result.get("columns"):
            result["columns"] = ["ROWID"]

        if result["is_related"] == "N":
            return {
                "input_item": {
                    "table_name": table_name,
                    "conversation": conversation,
                    "allowed_col_names": allowed_col_names
                },
                "filtered_schema": None,
                "error": None
            }
        else:
            return {
                "input_item": {
                    "table_name": table_name, 
                    "conversation": conversation, 
                    "allowed_col_names": allowed_col_names
                },
                "filtered_schema": (table_name, result["columns"]),
                "error": None
            }
    except Exception as e:
        return {
            "input_item": {
                "table_name": table_name, 
                "conversation": conversation, 
                "allowed_col_names": allowed_col_names
            },
            "filtered_schema": None,
            "error": str(e)
        }


async def link_schema(
    state: SQLAssistantState,
    chat_model: BaseChatModel,
    database: SQLiteDatabase,
) -> Dict[str, Dict[str, str]]:
    conversation = state.get("conversation")
    if not conversation:
        raise ValueError("conversation is required")
    max_retries = 1
    # queue = []
    # for table in  database.get_usable_table_names():
    #     for col_group in database.get_column_groups(table):
    #         queue.append({
    #             "table_name": table,
    #             "allowed_col_names": col_group,
    #             "conversation": conversation
    #         })
    queue = [
        {"table_name": table_name, "conversation": conversation} 
        for table_name in database.get_usable_table_names()
    ]
    successful_results = []
    for _ in range(max_retries):
        tasks = [_link_schema_one(chat_model=chat_model, database=database, **input_item) for input_item in queue]
        results = await asyncio.gather(*tasks)
        successful_results.extend([
            res for res in results if res["error"] is None
        ])
        failed_items = [
            res["input_item"] for res in results if res["error"] is not None
        ]
        queue = failed_items
        if not queue:
            break
    
    linked_schema = [
        result["filtered_schema"] 
        for result in successful_results 
        if result["filtered_schema"]
    ]
    # Return per-table mapping: column_name -> datatype
    final_schema: Dict[str, Dict[str, str]] = {}
    for table_name, col_names in linked_schema:
        table_schema = final_schema.setdefault(table_name, {})
        for col_name in col_names:
            col_type = database.get_column_datatype(
                table_name,
                col_name,
                default="NULL",
            )
            if col_type != "NULL":
                table_schema[col_name] = col_type

    state["linked_schema"] = final_schema
    return state