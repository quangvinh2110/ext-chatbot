from typing import TypedDict, Dict, List, Any, Optional
from langchain_core.messages import AnyMessage


class SQLAssistantState(TypedDict):
    conversation: List[AnyMessage]
    context: Optional[str]
    relevant_tables: List[str]
    sample_values: Dict[str, Dict[str, List[Any]]]
    linked_schema: Dict[str, Dict[str, str]]
    sql_queries: List[str]
    db_output: Dict[str, Any]
    relaxation_analysis: Optional[Dict[str, Any]]
    return_message: str