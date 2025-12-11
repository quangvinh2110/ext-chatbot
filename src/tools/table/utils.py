import sqlite3
import re
from typing import Dict, Any, List, Optional


def truncate_word(content: Any, *, length: int, suffix: str = "...") -> str:
    """Truncate a string to a certain number of words, based on the max string length."""
    if not isinstance(content, str) or length <= 0:
        return content
    if len(content) <= length:
        return content
    return content[: length - len(suffix)].rsplit(" ", 1)[0] + suffix




def pydantic_to_sqlite_type(pydantic_type: str) -> str:
    type_mapping = {
        'string': 'TEXT',
        'integer': 'INTEGER',
        'number': 'REAL',
        'boolean': 'INTEGER',  # SQLite uses INTEGER for booleans (0 or 1)
        'null': 'TEXT',
    }
    return type_mapping.get(pydantic_type.lower(), 'TEXT')


def create_sqlite_table(
    schema: Dict[str, Any],
    column_groups: List[List[str]],
    data: List[Dict[str, Any]],
    db_path: str,
    table_name: Optional[str] = None,
    if_exists: str = "replace"  # "replace", "append", or "fail"
) -> str:
    """
    Create a SQLite table from a Pydantic schema and import data.
    
    Args:
        schema: Pydantic schema in JSON schema format with 'title', 'type', and 'properties'
        column_groups: List of column groups
        data: List of dictionaries containing data to import
        db_path: Path to the SQLite database file
        table_name: Name of the table. If None, uses schema['title'] or 'RowData'
        if_exists: What to do if table exists: "replace" (drop and recreate), 
                  "append" (add to existing), or "fail" (raise error)
    
    Returns:
        The name of the created table
    
    Example:
        schema = {
            'title': 'RowData',
            'type': 'object',
            'properties': {
                'Loại thống kê': {'type': 'string', ...},
                'Giá trị': {'type': 'string', ...}
            }
        }
        data = [
            {'Giá trị': '500', 'Loại thống kê': 'Tổng số BĐS cho thuê'},
            ...
        ]
        create_sqlite_table(schema, data, 'database.db')
    """
    # Determine table name
    if table_name is None:
        table_name = schema.get('title', 'RowData')
    
    # Sanitize table name (SQLite allows most characters, but we'll quote it)
    # Remove or replace problematic characters
    # table_name = re.sub(r'[^\w\s]', '_', table_name)
    # table_name = re.sub(r'\s+', '_', table_name)
    
    # Build quick lookup for column -> group id
    column_groups = column_groups or []
    column_group_lookup: Dict[str, int] = {}
    for group_id, group in enumerate(column_groups):
        for col in group:
            # first group wins if duplicates
            column_group_lookup.setdefault(col, group_id)

    # Get properties from schema
    properties = schema.get('properties', {})
    if not properties:
        raise ValueError("Schema must contain 'properties'")
    
    # Build column definitions
    columns = []
    column_names = []
    for prop_name, prop_schema in properties.items():
        # Get type from schema
        prop_type = prop_schema.get('type', 'string')
        sqlite_type = pydantic_to_sqlite_type(prop_type)
        
        # Sanitize column name but keep original for data mapping
        safe_col_name = re.sub(r'[^\w\s]', '_', prop_name)
        safe_col_name = re.sub(r'\s+', '_', safe_col_name)
        
        # Use quoted identifiers to preserve original names if needed
        # SQLite supports quoted identifiers with square brackets or double quotes
        quoted_col_name = f'"{prop_name}"'
        
        columns.append(f'{quoted_col_name} {sqlite_type}')
        column_names.append(prop_name)
    
    # Create table SQL
    create_table_sql = '''
    CREATE TABLE IF NOT EXISTS "{table_name}" (\n\t{columns}\n)
    '''.format(
        table_name=table_name,
        columns=',\n\t'.join(columns)
    ).strip()

    # Metadata (EAV) table for column descriptions and groups
    metadata_table_name = f"{table_name}__metadata"
    create_metadata_table_sql = f'''
    CREATE TABLE IF NOT EXISTS "{metadata_table_name}" (
        entity TEXT NOT NULL,      -- column name
        attribute TEXT NOT NULL,   -- "description" | "group"
        value TEXT,
        PRIMARY KEY (entity, attribute)
    )
    '''.strip()
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Handle if_exists option
        if if_exists == "replace":
            cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            cursor.execute(f'DROP TABLE IF EXISTS "{metadata_table_name}"')
        elif if_exists == "append":
            # Check if table exists
            cursor.execute('''
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            ''', (table_name,))
            if cursor.fetchone():
                # Table exists, we'll append
                pass
            else:
                # Table doesn't exist, create it
                cursor.execute(create_table_sql)
            # Ensure metadata table exists for append path
            cursor.execute(create_metadata_table_sql)
        elif if_exists == "fail":
            # Check if table exists
            cursor.execute('''
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            ''', (table_name,))
            if cursor.fetchone():
                raise ValueError(f"Table '{table_name}' already exists")
            cursor.execute(create_table_sql)
            cursor.execute(create_metadata_table_sql)
        else:
            raise ValueError(f"Invalid if_exists value: {if_exists}. Must be 'replace', 'append', or 'fail'")
        
        # Create table if it doesn't exist (for append case where table might already exist)
        cursor.execute(create_table_sql)
        cursor.execute(create_metadata_table_sql)
        
        # Prepare insert statement
        placeholders = ', '.join(['?' for _ in column_names])
        quoted_column_names = ', '.join([f'"{col}"' for col in column_names])
        insert_sql = f'''
        INSERT INTO "{table_name}" ({quoted_column_names})
        VALUES ({placeholders})
        '''
        insert_metadata_sql = f'''
        INSERT OR REPLACE INTO "{metadata_table_name}" (entity, attribute, value)
        VALUES (?, ?, ?)
        '''
        
        # Insert data
        if data:
            rows_to_insert = []
            for row in data:
                # Extract values in the order of column_names
                values = [row.get(col_name) for col_name in column_names]
                rows_to_insert.append(values)
            
            cursor.executemany(insert_sql, rows_to_insert)

        # Insert metadata rows (one per column * attributes)
        metadata_rows = []
        for col_name, prop_schema in properties.items():
            description = prop_schema.get('description')
            group_idx: Optional[int] = column_group_lookup.get(col_name)
            metadata_rows.append((col_name, 'description', description))
            metadata_rows.append((col_name, 'group', None if group_idx is None else str(group_idx)))
        cursor.executemany(insert_metadata_sql, metadata_rows)
        
        conn.commit()
        return table_name
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
