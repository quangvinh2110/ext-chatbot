from typing import TypedDict, Dict, List, Any, Optional
from langchain_core.messages import AnyMessage


class SQLAssistantState(TypedDict):
    conversation: List[AnyMessage]
    linked_schema: Dict[str, Dict[str, str]]
    sql_queries: List[str]
    predicate_values: List[Dict[str, Any]]
    tbl_col_sample_values: Dict[str, Dict[str, List[Any]]]
    db_output: Dict[str, Any]
    final_answer: Optional[str]