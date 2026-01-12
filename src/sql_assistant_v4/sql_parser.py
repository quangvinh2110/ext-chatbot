from typing import List, Dict
import re
from sqlglot import parse_one, exp
import itertools
from collections import Counter
from sqlglot.optimizer.normalize import normalize

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
    sql_queries: List[str] = state.get("sql_queries", [])
    if not sql_queries:
        raise ValueError("SQL queries are required")
    sql_query = sql_queries[-1]
    if not sql_query:
        raise ValueError("SQL query is required")
    schema: Dict[str, Dict[str, str]] = state.get("linked_schema", {})
    if not schema:
        raise ValueError("Schema is required")
        
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


async def find_relaxed_constraints(
    state: SQLAssistantState,
    database: SQLiteDatabase,
    max_rows: int = 15,
) -> SQLAssistantState:
    sql_queries: List[str] = state.get("sql_queries", [])
    if not sql_queries:
        return state
    
    original_sql = sql_queries[-1]
    
    # 1. Parse and Get Real Tables
    try:
        parsed = parse_one(original_sql, read=database.dialect.lower())
    except Exception:
        return state

    # We need to know what the real tables are to decide if a subquery is "safe" to relax
    # (i.e., we are filtering raw data, not complex calculated CTEs)
    usable_tables = set(database.get_usable_table_names())

    # ---------------------------------------------------------
    # 2. Identify All WHERE scopes (Main, CTEs, Subqueries, Unions)
    # ---------------------------------------------------------
    # We find ALL exp.Where nodes in the entire tree
    all_where_nodes = list(parsed.find_all(exp.Where))
    
    if not all_where_nodes:
        return state

    best_result = None

    # ---------------------------------------------------------
    # 3. Iterate Over Each Scope
    # ---------------------------------------------------------
    for where_node in all_where_nodes:
        
        # 3a. Scope Safety Check
        # We try to identify the tables involved in this specific SELECT/UPDATE statement.
        # If this WHERE clause filters a table that isn't in our "usable_tables", 
        # it might be a complex CTE alias. We skip it to avoid breaking logic.
        parent_select = where_node.find_ancestor(exp.Select)
        if not parent_select: 
            continue # Should not happen in standard SQL

        scope_tables = set()
        for table in parent_select.find_all(exp.Table):
            # table.name is the real name, table.alias is the alias
            scope_tables.add(table.name)
        
        # If NO tables in this scope are real DB tables, assume it's a wrapper 
        # around a complex CTE and skip relaxing this specific node.
        # (We allow partial matches, e.g. JOINs between Real Table and CTE)
        if not scope_tables.intersection(usable_tables):
            # Exception: If it's a simple query with aliased tables not in usable list?
            # We assume usable_tables contains the raw table names.
            continue

        # ---------------------------------------------------------
        # 3b. Normalize & Group Constraints (DNF)
        # ---------------------------------------------------------
        # Copy the condition to work on it
        original_condition = where_node.this
        normalized_condition = normalize(original_condition)

        constraint_groups = []
        
        # Helper to flatten ANDs
        def get_and_constraints(node):
            if isinstance(node, exp.And):
                return get_and_constraints(node.this) + get_and_constraints(node.expression)
            return [node]

        # Extract branches (ORs)
        if isinstance(normalized_condition, exp.Or):
            def get_or_branches(node):
                if isinstance(node, exp.Or):
                    return get_or_branches(node.this) + get_or_branches(node.expression)
                return [node]
            branches = get_or_branches(normalized_condition)
            for branch in branches:
                constraint_groups.append(get_and_constraints(branch))
        else:
            constraint_groups.append(get_and_constraints(normalized_condition))

        # ---------------------------------------------------------
        # 3c. Attempt Relaxation on this Scope
        # ---------------------------------------------------------
        found_in_scope = False
        
        for group in constraint_groups:
            if len(group) < 2: 
                continue # Skip single conditions
            if found_in_scope: 
                break

            # Try removing 1..N constraints
            for r in range(len(group) - 1, 0, -1):
                if found_in_scope: 
                    break
                
                for subset in itertools.combinations(group, r):
                    # 1. Clone the FULL original parsed tree
                    # We must run the FULL query, just with this specific WHERE modified
                    temp_full_tree = parsed.copy()
                    
                    # 2. Locate the specific WHERE node in the cloned tree
                    # We cannot use the 'where_node' reference directly because it points to old tree.
                    # We find it by path or simply by matching structure? 
                    # Easier: We know the index of where_node in all_where_nodes.
                    # Let's find the corresponding node in temp_full_tree.
                    temp_where_nodes = list(temp_full_tree.find_all(exp.Where))
                    current_idx = all_where_nodes.index(where_node)
                    
                    if current_idx >= len(temp_where_nodes):
                        continue # Structure mismatch safety
                    
                    target_where = temp_where_nodes[current_idx]

                    # 3. Construct the subset condition
                    if len(subset) == 1:
                        new_cond = subset[0]
                    else:
                        new_cond = subset[0]
                        for i in range(1, len(subset)):
                            new_cond = exp.And(this=new_cond, expression=subset[i])

                    # 4. Swap it in
                    target_where.set("this", new_cond)
                    
                    # 5. Run it
                    modified_sql = temp_full_tree.sql(dialect=database.dialect.lower())
                    check_sql = f"{modified_sql} LIMIT {max_rows + 5}"
                    
                    results = await database.run_no_throw(check_sql, include_columns=True)
                    
                    if results and len(results) > 0:
                        found_in_scope = True
                        
                        # Calculate Drop List
                        subset_sqls = {n.sql() for n in subset}
                        dropped = [c for c in group if c.sql() not in subset_sqls]
                        
                        # Logic to prioritize the "Best" result across all scopes?
                        # Usually, deeper scopes (subqueries) returning data is better than 
                        # outer scopes returning data, but here we just take the first valid relaxation found.
                        # Or strictly compare counts/scores.
                        current_score = len(subset) # Prefer keeping MORE constraints
                        
                        if best_result is None or current_score > best_result["score"]:
                            best_result = {
                                "score": current_score,
                                "results": results,
                                "dropped_constraints": dropped,
                                "scope_tables": list(scope_tables),
                                "result_count": len(results)
                            }
                        break # Found best subset for this group

    # ---------------------------------------------------------
    # 4. Generate Analysis
    # ---------------------------------------------------------
    if not best_result:
        return state

    results = best_result["results"]
    dropped_nodes = best_result["dropped_constraints"]
    
    # Extract columns from dropped nodes for reporting
    cols_of_interest = set()
    for node in dropped_nodes:
        for col in node.find_all(exp.Column):
            cols_of_interest.add(col.name)

    dropped_str = ", ".join([n.sql() for n in dropped_nodes])
    
    # Format Data Stats
    stats_msgs = []
    if best_result["result_count"] > max_rows:
        for col_name in cols_of_interest:
            # Case-insensitive key lookup
            sample = results[0]
            key = next((k for k in sample.keys() if k.lower() == col_name.lower()), None)
            
            if key:
                vals = [r[key] for r in results if r[key] is not None]
                if vals:
                    if isinstance(vals[0], (int, float)):
                        stats_msgs.append(f"{col_name} range: {min(vals)} - {max(vals)}")
                    else:
                        # Top 5 distinct strings
                        top_k = [str(k) for k, v in Counter(vals).most_common(5)]
                        stats_msgs.append(f"{col_name} examples: {', '.join(top_k)}")
    
    stats_text = "; ".join(stats_msgs)
    
    # Construct Message
    if best_result["result_count"] > max_rows:
        msg = (
            f"No results found. However, inside the query logic for tables {best_result['scope_tables']}, "
            f"if we remove conditions [{dropped_str}], we find over {max_rows} matches. "
            f"Available data in that subset: {stats_text}. "
            f"Would you like to refine your filters?"
        )
    else:
        msg = (
            f"No results found. However, if we relax the conditions [{dropped_str}] "
            f"(specifically in the logic for {best_result['scope_tables']}), "
            f"I found {best_result['result_count']} results. Would you like to see them?"
        )

    state["relaxation_analysis"] = {
        "status": "relaxed_success",
        "dropped": [n.sql() for n in dropped_nodes],
        "message": msg,
        "subset_results": results if len(results) <= max_rows else []
    }
    
    return state