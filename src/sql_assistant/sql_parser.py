from sqlglot import exp, parse_one
from typing import List, Dict

from .pipeline import SQLAssistantState
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
    
    # --- Step A: Resolve Aliases (c -> customers) ---
    alias_map = {}
    
    # 1. Check FROM
    for from_node in parsed.find_all(exp.From):
        for table in from_node.find_all(exp.Table):
            real_name = table.name
            alias = table.alias if table.alias else real_name
            alias_map[alias] = real_name

    # 2. Check JOINs
    for join_node in parsed.find_all(exp.Join):
        table = join_node.this
        real_name = table.name
        alias = table.alias if table.alias else real_name
        alias_map[alias] = real_name

    print(f"DEBUG: Found Aliases: {alias_map}")

    extracted_data = []

    # --- Step B: Recursive Visitor ---
    def visit_node(node):
        if not node:
            return

        # 1. Handle Binary Logic (AND, OR)
        # sqlglot stores left side in 'this' and right side in 'expression'
        if isinstance(node, (exp.And, exp.Or)):
            visit_node(node.this)
            visit_node(node.expression)
            return

        # 2. Handle Wrappers (Parenthesis, NOT, WHERE)
        # These only have one child stored in 'this'
        if isinstance(node, (exp.Paren, exp.Not, exp.Where)):
            visit_node(node.this)
            return

        # 3. Handle Comparisons (Column = 'Value', !=, LIKE)
        if isinstance(node, (exp.EQ, exp.NEQ, exp.Like, exp.ILike)):
            # We look for: Column op Literal
            if isinstance(node.left, exp.Column) and isinstance(node.right, exp.Literal):
                if node.right.is_string:
                    process_extraction(node.left, node.right.this, node.key)
            return

        # 4. Handle IN (Column IN ('A', 'B'))
        if isinstance(node, exp.In):
            if isinstance(node.this, exp.Column):
                # The list of values is in args['expressions']
                for item in node.args.get('expressions', []):
                    if isinstance(item, exp.Literal) and item.is_string:
                        process_extraction(node.this, item.this, "IN")
            return

    # Helper to validate and store
    def process_extraction(col_node, value_str, operator):
        col_name = col_node.name
        table_alias = col_node.table
        
        real_table_name = None

        # Resolve Alias
        if table_alias:
            real_table_name = alias_map.get(table_alias)
        else:
            # Try to guess table from schema if no alias provided
            matches = [t for t, cols in schema.items() if col_name in cols]
            if len(matches) == 1:
                real_table_name = matches[0]

        # Validation
        if real_table_name and real_table_name in schema:
            cols = schema[real_table_name]
            if col_name in cols:
                if cols[col_name] == "TEXT":
                    extracted_data.append({
                        "table_name": real_table_name,
                        "column_name": col_name,
                        "value": value_str,
                        "operator": operator
                    })
                else:
                    print(f"DEBUG: Skipped {col_name} (Not TEXT)")
            else:
                print(f"DEBUG: Skipped {col_name} (Not in {real_table_name})")
        else:
            print(f"DEBUG: Skipped {col_name} (Unknown table/alias)")

    # --- Step C: Start Traversal ---
    where_clause = parsed.find(exp.Where)
    if where_clause:
        # Crucial Fix: Pass where_clause.this (the content) OR rely on the updated visitor handling exp.Where
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