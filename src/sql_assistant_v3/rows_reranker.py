from typing import Dict, List, Any
import asyncio
import json

from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .state import SQLAssistantState
from ..utils import format_conversation
from ..tools.sqlite_database import SQLiteDatabase
from ..prompts import RERANK_ROWS_TEMPLATE

_rows_reranking_chain_cache: Dict[int, Runnable] = {}
def get_rows_reranking_chain(chat_model: BaseChatModel) -> Runnable:
    # Use model instance ID as cache key (since ChatOpenAI objects aren't hashable)
    chat_model_id = id(chat_model)
    
    if chat_model_id not in _rows_reranking_chain_cache:
        _rows_reranking_chain_cache[chat_model_id] = (
            ChatPromptTemplate([("human", RERANK_ROWS_TEMPLATE)])
            | chat_model
            | JsonOutputParser()
        )
    
    return _rows_reranking_chain_cache[chat_model_id]


def iter_batch(_iterable: list, batch_size: int):
    for i in range(0, len(_iterable), batch_size):
        yield _iterable[i:i+batch_size]


async def _rerank_one(
    conversation: List[AnyMessage],
    table_name: str,
    allowed_col_names: List[str],
    column_sample_values: Dict[str, List[str]],
    row_ids: List[int],
    chat_model: BaseChatModel,
    database: SQLiteDatabase,
) -> Dict[str, Any]:
    try:
        
        rows = await database.run_no_throw(
            "SELECT rowid, {column_names} FROM \"{table_name}\" WHERE rowid IN ({row_ids});".format(
                column_names=', '.join(f'"{col}"' for col in allowed_col_names),
                table_name=table_name,
                row_ids=', '.join(map(str, row_ids))
            ),
            include_columns=True
        )
        formatted_rows = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows["result"])
        formatted_conversation = format_conversation(conversation)
        table_info = database.get_table_info_no_throw(
            table_name,
            get_col_comments=True,
            allowed_col_names=allowed_col_names,
            column_sample_values=column_sample_values,
            sample_count=5
        )

        result = await get_rows_reranking_chain(chat_model).ainvoke({
            "table_name": table_name, 
            "table_info": table_info, 
            "formatted_conversation": formatted_conversation, 
            "formatted_rows": formatted_rows
        })

        if "relevant_row_ids" not in result:
            raise ValueError("Invalid response from rows re-ranking chain")
        
        relevant_row_ids = list(set(result["relevant_row_ids"]))
        if len(relevant_row_ids) > len(row_ids):
            raise ValueError("More relevant rows than candidate rows")
        for rowid in relevant_row_ids:
            if rowid not in row_ids:
                raise ValueError(f"Row ID {rowid} not in candidate rows")

        return {
            "input_item": {
                "table_name": table_name,
                "conversation": conversation,
                "row_ids": row_ids,
            },
            "filtered_row_ids": relevant_row_ids,
            "error": None
        }
    except Exception as e:
        return {
            "input_item": {
                "table_name": table_name,
                "conversation": conversation,
                "row_ids": row_ids,
            },
            "filtered_row_ids": None,
            "error": str(e)
        }


async def rerank_rows(
    state: SQLAssistantState,
    chat_model: BaseChatModel,
    database: SQLiteDatabase,
) -> Dict[str, Dict[str, str]]:
    conversation = state.get("conversation")
    if not conversation:
        raise ValueError("conversation is required")
    linked_schema = state.get("linked_schema")
    if not linked_schema:
        filtered_tables = database.get_usable_table_names()
    else:   
        filtered_tables = list(linked_schema.keys())
    sample_values = state.get("sample_values", {})
    max_retries = 1
    queue = []
    for table_name in filtered_tables:
        all_row_ids = await database.run_no_throw(f"select rowid from \"{table_name}\"")
        for batch_row_ids in iter_batch(all_row_ids["result"], 3):
            queue.append({
                "table_name": table_name,
                "conversation": conversation,
                "row_ids": [item[0] for item in batch_row_ids],
                "allowed_col_names": list(linked_schema.get(table_name, {}).keys()),
                "column_sample_values": sample_values.get(table_name, {}),
            })
    successful_results = []
    for _ in range(max_retries):
        tasks = [_rerank_one(chat_model=chat_model, database=database, **input_item) for input_item in queue]
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
    
    filtered_row_ids = []
    for result in successful_results:
        if result.get("filtered_row_ids"):
            filtered_row_ids.extend(result["filtered_row_ids"])
    filtered_row_ids = list(set(filtered_row_ids))
    filtered_rows = await database.run_no_throw(
        f"SELECT rowid, * FROM \"{table_name}\" WHERE rowid IN ({', '.join(map(str, filtered_row_ids))});",
        include_columns=True
    )

    state["db_output"] = {
        "result": filtered_rows["result"],
        "error": None
    }
    return state