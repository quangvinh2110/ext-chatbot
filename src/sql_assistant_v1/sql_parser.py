from typing import List, Dict
import re
from sqlglot import parse_one, exp

from ..tools.table.sqlite_database import SQLiteDatabase
from .state import SQLAssistantState


def restrict_select_columns(
    state: SQLAssistantState,
    database: SQLiteDatabase,
) -> SQLAssistantState:
    """
    Replaces SELECT * with SELECT t.col1, t.col2 based on filtered_schema.
    """
    sql_queries: List[str] = state.get("sql_queries")
    if not sql_queries:
        raise ValueError("SQL queries are required")
    sql_query = sql_queries[-1]
    if not sql_query:
        raise ValueError("SQL query is required")
    schema: Dict[str, Dict[str, str]] = state.get("linked_schema")
    if not schema:
        raise ValueError("Schema is required")
    parsed = parse_one(sql_query, read=database.dialect.lower())
    
    # ---------------------------------------------------------
    # 1. Build Alias Map (Map Alias -> Real Table Name)
    # ---------------------------------------------------------
    # We need to know the order of tables to expand * correctly
    active_tables_ordered = [] 
    alias_map = {}

    def register_table(table_node):
        real_name = table_node.name
        alias = table_node.alias if table_node.alias else real_name
        
        # Only register if we haven't seen this alias yet
        if alias not in alias_map:
            alias_map[alias] = real_name
            active_tables_ordered.append(alias)

    # Scan FROM
    for from_node in parsed.find_all(exp.From):
        for table in from_node.find_all(exp.Table):
            register_table(table)

    # Scan JOINs
    for join_node in parsed.find_all(exp.Join):
        register_table(join_node.this)

    print(f"DEBUG: Active Tables: {alias_map}")

    # ---------------------------------------------------------
    # 2. Helper to Generate Column Expressions
    # ---------------------------------------------------------
    def get_columns_for_table(table_alias):
        real_name = alias_map.get(table_alias)
        if not real_name or real_name not in schema:
            return [] # Table not in our allowed schema, return nothing (or handle error)
        
        # Create sqlglot Column objects: alias.column_name
        cols = schema[real_name].keys()
        return [
            exp.Column(
                this=exp.Identifier(this=col, quoted=True),
                table=exp.Identifier(this=table_alias, quoted=True)
            ) for col in cols
        ]

    # ---------------------------------------------------------
    # 3. Rewrite SELECT Expressions
    # ---------------------------------------------------------
    # We only want to transform the main SELECT statement(s)
    for select_node in parsed.find_all(exp.Select):
        new_expressions = []
        
        for expression in select_node.expressions:
            # Case A: Naked * (SELECT *)
            if isinstance(expression, exp.Star) and not isinstance(expression, exp.Count):
                # Expand columns for ALL active tables in the query
                for alias in active_tables_ordered:
                    expanded_cols = get_columns_for_table(alias)
                    new_expressions.extend(expanded_cols)
            
            # Case B: Qualified * (SELECT t.*)
            elif isinstance(expression, exp.Column) and isinstance(expression.this, exp.Star):
                # Extract the table alias (e.g., 't' from 't.*')
                table_alias = expression.table
                expanded_cols = get_columns_for_table(table_alias)
                new_expressions.extend(expanded_cols)
                
            # Case C: Regular column or other expression (Keep it)
            else:
                new_expressions.append(expression)

        # Replace the old expressions with the new expanded list
        if new_expressions:
            select_node.set("expressions", new_expressions)

    restricted_sql_query = parsed.sql(dialect=database.dialect.lower())
    def normalize_sql_query(sql_query: str) -> str:
        return re.sub(r"\s+", " ", sql_query).strip().strip(";").lower()
    if normalize_sql_query(restricted_sql_query) != normalize_sql_query(sql_query):
        state["sql_queries"].append(restricted_sql_query)
    return state