import sqlite3
import faiss
import numpy as np
import re
import os
import json
from typing import Dict, Any, List, Optional, Iterable
from tqdm import tqdm

from .client import get_infinity_embeddings


def _safe_filename(name: str) -> str:
    """Make a reasonably safe filename from table/column names."""
    # Keep unicode but remove path separators and problematic chars
    name = name.replace(os.sep, "_").replace("\x00", "_")
    name = re.sub(r"[<>:\"/\\|?*\n\r\t]+", "_", name).strip()
    return name or "unnamed"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def create_sqlite(
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
    
    # Debug: Check if properties already has duplicates (shouldn't be possible but check anyway)
    print(f"Original properties count: {len(properties)}")
    print(f"Original property names: {list(properties.keys())}")
    
    # Handle duplicate column names (case-insensitive to match SQLite behavior)
    # Prioritize non-string types over string types
    deduplicated_properties: Dict[str, Any] = {}
    # Track lowercase names to their canonical (first seen) name
    lowercase_to_canonical: Dict[str, str] = {}
    
    for prop_name, prop_schema in properties.items():
        prop_name_lower = prop_name.lower()
        
        if prop_name_lower in lowercase_to_canonical:
            # Case-insensitive duplicate found
            canonical_name = lowercase_to_canonical[prop_name_lower]
            existing_type = deduplicated_properties[canonical_name].get('type', 'string')
            new_type = prop_schema.get('type', 'string')
            
            print(f"Warning: Case-insensitive duplicate column found: '{canonical_name}' vs '{prop_name}'")
            print(f"  Existing type: {existing_type}, New type: {new_type}")
            
            # Keep non-string type over string type
            if existing_type == 'string' and new_type != 'string':
                # Replace with new property but keep the canonical name
                deduplicated_properties[canonical_name] = prop_schema
                print(f"  -> Replacing with new property (keeping name '{canonical_name}')")
            else:
                print(f"  -> Keeping existing property '{canonical_name}'")
        else:
            # New property (no case-insensitive duplicate)
            lowercase_to_canonical[prop_name_lower] = prop_name
            deduplicated_properties[prop_name] = prop_schema
    
    # Build column definitions
    columns = []
    column_names = []
    seen_column_names = set()
    
    for prop_name, prop_schema in deduplicated_properties.items():
        # Skip if we've already added this column name
        if prop_name in seen_column_names:
            print(f"Warning: Skipping duplicate column '{prop_name}' in SQL generation")
            continue
        
        seen_column_names.add(prop_name)
        
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
    
    # Debug: Check for duplicates in column_names
    if len(column_names) != len(set(column_names)):
        from collections import Counter
        duplicates = [name for name, count in Counter(column_names).items() if count > 1]
        print(f"ERROR: Duplicate column names found: {duplicates}")
        print(f"All column names: {column_names}")
    
    # Create table SQL
    create_table_sql = '''
    CREATE TABLE IF NOT EXISTS "{table_name}" (\n\t{columns}\n)
    '''.format(
        table_name=table_name,
        columns=',\n\t'.join(columns)
    ).strip()
    
    # Debug: Print the SQL being generated
    print(f"Creating table: {table_name}")
    print(f"Number of columns: {len(column_names)}")
    if len(column_names) != len(set(column_names)):
        print(f"SQL Statement:\n{create_table_sql}")

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
                values = []
                for col_name in column_names:
                    value = row.get(col_name)
                    # Handle list values by joining with "; "
                    if isinstance(value, list):
                        value = "; ".join(str(item) for item in value)
                    values.append(value)
                rows_to_insert.append(values)
            
            cursor.executemany(insert_sql, rows_to_insert)

        # Insert metadata rows (one per column * attributes)
        metadata_rows = []
        for col_name, prop_schema in deduplicated_properties.items():
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


def _get_text_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    cursor = conn.cursor()
    cursor.execute(f'PRAGMA table_info("{table_name}")')
    cols = cursor.fetchall()
    # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
    text_cols: List[str] = []
    for _, col_name, col_type, *_rest in cols:
        if not col_type:
            continue
        if "TEXT" in str(col_type).upper():
            text_cols.append(col_name)
    return text_cols


def _iter_text_values(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    *,
    distinct: bool = True,
) -> Iterable[str]:
    cursor = conn.cursor()
    distinct_sql = "DISTINCT " if distinct else ""
    cursor.execute(
        f'SELECT {distinct_sql}"{column_name}" FROM "{table_name}" WHERE "{column_name}" IS NOT NULL'
    )
    for (val,) in cursor.fetchall():
        # Keep as string for embedding
        if val is None:
            continue
        yield str(val)

        

def create_faiss(
    schema: Dict[str, Any],
    db_path: str,
    faiss_dir: str,
    table_name: Optional[str] = None,
    *,
    if_exists: str = "replace",  # "replace", "append", or "fail"
    embed_model: str,
    embed_api_url: str,
    batch_size: int = 128,
    distinct_values: bool = True,
    normalize: bool = True,
    max_text_length: int = 512,
) -> Dict[str, Any]:
    """
    Build FAISS indexes for every TEXT column in the given SQLite table.

    The signature intentionally mirrors `create_sqlite_table(...)` so you can call it
    right after creating/importing the table.

    Output layout (per TEXT column):
      {faiss_dir}/{db_stem}/{table_name}/{column_name}.faiss
      {faiss_dir}/{db_stem}/{table_name}/{column_name}.json   (list of values aligned to vectors)

    Returns:
        Summary dict with created indexes and counts.

    Embeddings:
        Uses `langchain_community.embeddings.InfinityEmbeddings`. Pass `embed_api_url`
        (typically ending with `/v1`) or set `INFINITY_API_URL`.
    """
    # Determine table name (same rule as create_sqlite_table)
    if table_name is None:
        table_name = schema.get("title", "RowData")

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    db_stem = os.path.splitext(os.path.basename(db_path))[0] or "database"
    table_dir = os.path.join(faiss_dir, _safe_filename(db_stem), _safe_filename(table_name))
    _ensure_dir(table_dir)

    conn = sqlite3.connect(db_path)
    try:
        if not embed_api_url:
            raise ValueError(
                "Missing embed_api_url. Pass create_faiss(..., embed_api_url=...) "
                "or set INFINITY_API_URL (e.g. http://localhost:7797/v1)."
            )

        embeddings = get_infinity_embeddings(model=embed_model, infinity_api_url=embed_api_url)
        text_cols = _get_text_columns(conn, table_name)
        created: List[Dict[str, Any]] = []

        for col in text_cols:
            index_path = os.path.join(table_dir, f"{_safe_filename(col)}.faiss")
            values_path = os.path.join(table_dir, f"{_safe_filename(col)}.json")

            if if_exists == "fail" and (os.path.exists(index_path) or os.path.exists(values_path)):
                raise FileExistsError(f"FAISS artifacts already exist for {table_name}.{col}")

            existing_values: List[str] = []
            existing_set: set[str] = set()
            index = None

            if if_exists == "replace":
                # Remove any previous artifacts for this column
                for p in (index_path, values_path):
                    if os.path.exists(p):
                        os.remove(p)
            elif if_exists == "append":
                if os.path.exists(values_path):
                    with open(values_path, "r", encoding="utf-8") as f:
                        existing_values_raw = json.load(f) or []
                    if not isinstance(existing_values_raw, list):
                        raise ValueError(f"Unexpected mapping file format: {values_path}")
                    # Normalize existing values: strip prefix if present (we store only values)
                    prefix = f"{col}: "
                    existing_values = []
                    existing_set.clear()  # Clear and rebuild for checking formatted values
                    for existing_val in existing_values_raw:
                        val_str = str(existing_val)
                        # Strip prefix if present (migration from old format)
                        if val_str.startswith(prefix):
                            val_str = val_str[len(prefix):]
                        existing_values.append(val_str)
                        # Add formatted version to set for duplicate checking
                        existing_set.add(f"{prefix}{val_str}")
                if os.path.exists(index_path):
                    index = faiss.read_index(index_path)
            elif if_exists not in ("replace", "append", "fail"):
                raise ValueError(
                    f"Invalid if_exists value: {if_exists}. Must be 'replace', 'append', or 'fail'"
                )

            # Collect values to embed (format: "column_name: column_value")
            # But store only column values (without prefix) in JSON
            values_to_embed: List[str] = []  # Formatted for embedding
            values_to_store: List[str] = []  # Unformatted for storage
            prefix = f"{col}: "
            prefix_len = len(prefix)
            for v in _iter_text_values(conn, table_name, col, distinct=distinct_values):
                v = v.strip()
                if not v:
                    continue
                # Truncate value part to account for prefix length
                if max_text_length and len(v) > (max_text_length - prefix_len):
                    v = v[:max_text_length - prefix_len]
                # Format as "column_name: column_value" for embedding
                formatted_value = f"{prefix}{v}"
                if existing_set and formatted_value in existing_set:
                    continue
                values_to_embed.append(formatted_value)
                values_to_store.append(v)  # Store only the value

            # Nothing new to add
            if not values_to_embed and index is not None and existing_values:
                created.append(
                    {
                        "table": table_name,
                        "column": col,
                        "index_path": index_path,
                        "values_path": values_path,
                        "added": 0,
                        "total": len(existing_values),
                    }
                )
                continue

            # Initialize index if needed (we'll determine dim from first batch)
            d = None
            total_added = 0

            # Embed in batches and add incrementally to avoid memory issues
            # This way we only hold one batch in RAM at a time
            with tqdm(
                total=len(values_to_embed),
                desc=f"Embedding {table_name}.{col}",
                unit="values"
            ) as pbar:
                for i in range(0, len(values_to_embed), batch_size):
                    batch = values_to_embed[i : i + batch_size]
                    
                    # Embed this batch
                    batch_vectors = embeddings.embed_documents(batch)
                    
                    if not batch_vectors:
                        pbar.update(len(batch))
                        continue
                    
                    # Convert to numpy array
                    vectors = np.asarray(batch_vectors, dtype="float32")
                    if vectors.ndim != 2:
                        raise ValueError(f"Unexpected embedding matrix shape: {vectors.shape}")
                    
                    # Determine dimension from first batch
                    if d is None:
                        d = int(vectors.shape[1])
                        if index is None:
                            index = faiss.IndexFlatIP(d) if normalize else faiss.IndexFlatL2(d)
                    elif vectors.shape[1] != d:
                        raise ValueError(
                            f"Embedding dimension mismatch: expected {d}, got {vectors.shape[1]}"
                        )
                    
                    # Normalize if needed
                    if normalize:
                        faiss.normalize_L2(vectors)
                    
                    # Add to index immediately (incremental, memory-efficient)
                    if index is None:
                        raise RuntimeError("FAISS index was not initialized before adding vectors")
                    index.add(vectors)
                    total_added += len(batch)
                    pbar.update(len(batch))

            # Build final values list (store only column values, not formatted)
            final_values = existing_values + values_to_store

            # Save index and mapping if we have an index (either loaded or newly created)
            if index is not None:
                # Persist artifacts
                faiss.write_index(index, index_path)
                with open(values_path, "w", encoding="utf-8") as f:
                    json.dump(final_values, f, ensure_ascii=False)
            elif total_added == 0 and not existing_values:
                # No index was created and no existing values - save empty mapping only
                with open(values_path, "w", encoding="utf-8") as f:
                    json.dump([], f, ensure_ascii=False)
            
            created.append(
                {
                    "table": table_name,
                    "column": col,
                    "index_path": index_path,
                    "values_path": values_path,
                    "added": total_added,
                    "total": len(final_values),
                    "dim": d if d is not None else None,
                    "normalize": normalize,
                    "distinct_values": distinct_values,
                    "embed_model": embed_model,
                }
            )

        return {
            "db_path": db_path,
            "table_name": table_name,
            "faiss_dir": table_dir,
            "text_columns": text_cols,
            "indexes": created,
        }
    finally:
        conn.close()
