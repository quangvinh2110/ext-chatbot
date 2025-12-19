from typing import TypedDict, Dict, List, Any, Optional
from langchain_core.messages import AnyMessage


class SQLAssistantState(TypedDict):
    conversation: List[AnyMessage]
    rewritten_message: str
    sample_values: Dict[str, Dict[str, List[Any]]]
    linked_schema: Dict[str, Dict[str, str]]
    sql_queries: List[str]
    db_output: Dict[str, Any]
    final_answer: Optional[str]