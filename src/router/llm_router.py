from typing import Dict, List

from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .utils import format_conversation
from ..tools.table.sqlite_database import SQLiteDatabase
from ..prompts import ROUTING_TEMPLATE
from ..utils import parse_json_output


def extract_choice(msg_content: str) -> str:
    try:
        return parse_json_output(msg_content)["choice"]
    except Exception:
        choice_s = '"choice": "'
        i = msg_content.find(choice_s)
        if i > 0:
            choice_text = msg_content[i + len(choice_s):]
            return choice_text.split('"')[0]
        return "vector"


_router_chain_cache: Dict[int, Runnable] = {}
def get_router_chain(chat_model: BaseChatModel) -> Runnable:
    # Use model instance ID as cache key (since ChatOpenAI objects aren't hashable)
    chat_model_id = id(chat_model)

    if chat_model_id not in _router_chain_cache:
        _router_chain_cache[chat_model_id] = (
            ChatPromptTemplate(["human", ROUTING_TEMPLATE])
            | chat_model
            | StrOutputParser()
        )

    return _router_chain_cache[chat_model_id]


async def route(
    conversation: List[AnyMessage],
    chat_model: BaseChatModel,
    database: SQLiteDatabase,
) -> str:
    """
    Route the conversation to the appropriate data source.
    """
    formatted_conversation = format_conversation(conversation)
    table_overview = database.get_table_overview()
    sql_summaries = "\n".join(str({
        "name": table["name"],
        "summary": table["summary"]
    }) for table in table_overview if table["data_source"] == "sql")
    vector_summaries = "\n".join(str({
        "name": table["name"],
        "summary": table["summary"]
    }) for table in table_overview if table["data_source"] == "vector")
    result = await get_router_chain(chat_model).ainvoke({
        "sql_summaries": sql_summaries,
        "vector_summaries": vector_summaries,
        "formatted_conversation": formatted_conversation
    })
    return extract_choice(result)