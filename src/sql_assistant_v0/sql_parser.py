from typing import List, Dict
import re
from sqlglot import parse_one, exp

from .state import SQLAssistantState
from ..tools.sqlite_database import SQLiteDatabase


def refine_sql_query(
    state: SQLAssistantState,
    database: SQLiteDatabase,
) -> SQLAssistantState:
    """
    1. Replaces SELECT * with explicit columns based on schema.
    2. Normalizes WHERE clauses for TEXT columns to be case-insensitive 
       (e.g., col = 'Val' -> LOWER(col) = 'val').
    """
    sql_queries: List[str] = state.get("sql_queries")
    if not sql_queries:
        raise ValueError("SQL queries are required")
    sql_query = sql_queries[-1]
    if not sql_query:
        raise ValueError("SQL query is required")
    schema: Dict[str, Dict[str, str]] = {}
    for table_name in database.get_usable_table_names():
        schema[table_name] = {}
        for col_name in database.get_column_names(table_name):
            schema[table_name][col_name] = database.get_column_datatype(table_name, col_name)
    parsed = parse_one(sql_query, read=database.dialect.lower())
    
    # ---------------------------------------------------------
    # 1. Build Alias Map (Map Alias -> Real Table Name)
    # ---------------------------------------------------------
    active_tables_ordered = [] 
    alias_map = {}

    def register_table(table_node):
        real_name = table_node.name
        alias = table_node.alias if table_node.alias else real_name
        
        if alias not in alias_map:
            alias_map[alias] = real_name
            active_tables_ordered.append(alias)

    for from_node in parsed.find_all(exp.From):
        for table in from_node.find_all(exp.Table):
            register_table(table)

    for join_node in parsed.find_all(exp.Join):
        register_table(join_node.this)

    # ---------------------------------------------------------
    # 2. Rewrite SELECT Expressions (Expand *)
    # ---------------------------------------------------------
    def get_columns_for_table(table_alias):
        real_name = alias_map.get(table_alias)
        if not real_name or real_name not in schema:
            return []
        
        cols = schema[real_name].keys()
        return [
            exp.Column(
                this=exp.Identifier(this=col, quoted=True),
                table=exp.Identifier(this=table_alias, quoted=True)
            ) for col in cols
        ]

    for select_node in parsed.find_all(exp.Select):
        new_expressions = []
        for expression in select_node.expressions:
            if isinstance(expression, exp.Star) and not isinstance(expression, exp.Count):
                for alias in active_tables_ordered:
                    new_expressions.extend(get_columns_for_table(alias))
            elif isinstance(expression, exp.Column) and isinstance(expression.this, exp.Star):
                table_alias = expression.table
                new_expressions.extend(get_columns_for_table(table_alias))
            else:
                new_expressions.append(expression)

        if new_expressions:
            select_node.set("expressions", new_expressions)

    # ---------------------------------------------------------
    # 3. Helper: Resolve Table for a Column (Used for Type Checking)
    # ---------------------------------------------------------
    def resolve_real_table_name(col_node):
        col_name = col_node.name
        table_alias = col_node.table
        
        # Case A: Explicit Alias
        if table_alias:
            return alias_map.get(table_alias)
        
        # Case B: Implicit Alias (Search all active tables)
        candidates = []
        for alias, real_name in alias_map.items():
            if real_name in schema and col_name in schema[real_name]:
                candidates.append(real_name)
        
        if len(candidates) == 1:
            return candidates[0]
        return None

    # ---------------------------------------------------------
    # 4. Rewrite WHERE/HAVING to be Case Insensitive
    # ---------------------------------------------------------
    def make_case_insensitive(node):
        # Recursively visit children first (Depth-First)
        # This handles nested AND/OR structures automatically
        if isinstance(node, (exp.And, exp.Or, exp.Paren, exp.Not, exp.Where)):
            for child_key in node.arg_types:
                child = node.args.get(child_key)
                if isinstance(child, list):
                    for item in child:
                        make_case_insensitive(item)
                elif child:
                    make_case_insensitive(child)
            return

        # Handle Binary Comparisons (=, !=, LIKE)
        if isinstance(node, (exp.EQ, exp.NEQ, exp.Like)):
            left = node.left
            right = node.right
            
            # Check structure: Column op Literal string
            if isinstance(left, exp.Column) and isinstance(right, exp.Literal) and right.is_string:
                real_table = resolve_real_table_name(left)
                if real_table:
                    col_type = schema[real_table].get(left.name)
                    # Only apply to TEXT columns
                    if col_type == "TEXT":
                        # Apply LOWER() to the column side
                        # We use exp.Lower wrapping the existing column node
                        new_left = exp.Lower(this=left.copy())
                        
                        # Lowercase the string literal value
                        new_right = exp.Literal.string(right.this.lower())
                        
                        # Replace in the AST
                        node.set("this", new_left)
                        node.set("expression", new_right)

        # Handle IN clause (Column IN ('A', 'B'))
        if isinstance(node, exp.In) and isinstance(node.this, exp.Column):
            real_table = resolve_real_table_name(node.this)
            if real_table:
                col_type = schema[real_table].get(node.this.name)
                if col_type == "TEXT":
                    # 1. Lowercase the column
                    new_left = exp.Lower(this=node.this.copy())
                    node.set("this", new_left)
                    
                    # 2. Lowercase all literal arguments in the list
                    new_args = []
                    for arg in node.args.get("expressions", []):
                        if isinstance(arg, exp.Literal) and arg.is_string:
                            new_args.append(exp.Literal.string(arg.this.lower()))
                        else:
                            new_args.append(arg)
                    
                    node.set("expressions", new_args)

    # Apply to WHERE clause
    where_clause = parsed.find(exp.Where)
    if where_clause:
        make_case_insensitive(where_clause)

    # Apply to HAVING clause (optional, but good practice)
    having_clause = parsed.find(exp.Having)
    if having_clause:
        make_case_insensitive(having_clause)

    # ---------------------------------------------------------
    # 5. Finalize
    # ---------------------------------------------------------
    refined_sql = parsed.sql(dialect=database.dialect.lower())
    
    def normalize_sql_query(q: str) -> str:
        return re.sub(r"\s+", " ", q).strip().strip(";").lower()

    # Only append if changes were made
    if normalize_sql_query(refined_sql) != normalize_sql_query(sql_query):
        state["sql_queries"].append(refined_sql)
        
    return state