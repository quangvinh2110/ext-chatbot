from typing import TypedDict, Dict, List, Any
from langchain_core.messages import AnyMessage


class SQLAssistantState(TypedDict):
    conversation: List[AnyMessage]
    context: str
    sample_values: Dict[str, Dict[str, List[Any]]]
    sql_queries: List[str]
    db_output: Dict[str, Any]