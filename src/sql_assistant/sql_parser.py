import re
from sqlglot import exp, parse_one
from typing import List, Dict

from .full_pipeline import SQLAssistantState
from ..tools.table.sqlite_database import SQLiteDatabase


def get_predicate_values(
    state: SQLAssistantState,
    database: SQLiteDatabase,
) -> SQLAssistantState:
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
    # 1. Map Aliases AND Track Active Tables
    # ---------------------------------------------------------
    alias_map = {}
    
    # Helper to register tables found in FROM/JOIN
    def register_table(table_node):
        real_name = table_node.name
        alias = table_node.alias if table_node.alias else real_name
        alias_map[alias] = real_name

    for from_node in parsed.find_all(exp.From):
        for table in from_node.find_all(exp.Table):
            register_table(table)

    for join_node in parsed.find_all(exp.Join):
        register_table(join_node.this)

    print(f"DEBUG: Found Aliases: {alias_map}")

    extracted_data = []

    # ---------------------------------------------------------
    # 2. Logic to Resolve Table for a Column
    # ---------------------------------------------------------
    def resolve_table(col_node):
        col_name = col_node.name
        table_alias = col_node.table
        
        # Case A: Alias is explicit (e.g., c.country)
        if table_alias:
            return alias_map.get(table_alias)
        
        # Case B: No alias (e.g., country). 
        # FIX: Check only tables present in the current query (alias_map.values())
        active_tables = set(alias_map.values())
        
        candidates = []
        for table in active_tables:
            # Check if table exists in schema AND column exists in that table
            if table in schema and col_name in schema[table]:
                candidates.append(table)
        
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1:
            print(f"DEBUG: Ambiguous column '{col_name}' found in multiple active tables: {candidates}")
            return None
        else:
            return None

    # ---------------------------------------------------------
    # 3. Recursive Visitor
    # ---------------------------------------------------------
    def visit_node(node):
        if not node: 
            return

        if isinstance(node, (exp.And, exp.Or)):
            visit_node(node.this)
            visit_node(node.expression)
            return

        if isinstance(node, (exp.Paren, exp.Not, exp.Where)):
            visit_node(node.this)
            return

        # Handle Binary Comparisons (=, !=, LIKE)
        if isinstance(node, (exp.EQ, exp.NEQ, exp.Like, exp.ILike)):
            if isinstance(node.left, exp.Column) and isinstance(node.right, exp.Literal):
                if node.right.is_string:
                    process_extraction(node.left, node.right.this, node.key)
            return

        # Handle IN clause
        if isinstance(node, exp.In) and isinstance(node.this, exp.Column):
            for item in node.args.get('expressions', []):
                if isinstance(item, exp.Literal) and item.is_string:
                    process_extraction(node.this, item.this, "IN")
            return

    def process_extraction(col_node, value_str, operator):
        col_name = col_node.name
        real_table_name = resolve_table(col_node)

        if real_table_name:
            # Verify data type is TEXT
            col_type = schema[real_table_name].get(col_name)
            if col_type == "TEXT":
                extracted_data.append({
                    "table_name": real_table_name,
                    "column_name": col_name,
                    "value": value_str,
                    "operator": operator
                })
            else:
                print(f"DEBUG: Skipped {col_name} (Type is {col_type}, not TEXT)")
        else:
            print(f"DEBUG: Skipped {col_name} (Could not resolve table)")

    # ---------------------------------------------------------
    # 4. Execution
    # ---------------------------------------------------------
    where_clause = parsed.find(exp.Where)
    if where_clause:
        visit_node(where_clause)
    state["predicate_values"] = extracted_data
    return state


async def get_similar_predicate_values(
    state: SQLAssistantState,
    database: SQLiteDatabase,
) -> SQLAssistantState:
    predicate_values = state.get("predicate_values")
    if not predicate_values:
        state["tbl_col_sample_values"] = {}
        return state
    state["tbl_col_sample_values"] = await database.batch_search_similar_values(
        [
            (v["table_name"], v["column_name"], v["value"])
            for v in predicate_values
        ], 
        k=5
    )
    return state


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
        return re.sub(r"\s+", " ", sql_query).strip().lower()
    if normalize_sql_query(restricted_sql_query) != normalize_sql_query(sql_query):
        state["sql_queries"].append(restricted_sql_query)
    return state