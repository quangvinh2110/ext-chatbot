import json
import os
import asyncio
from typing import Any, Dict, Iterable, List, Literal, Sequence, Tuple, Union, Optional

import faiss
import numpy as np
from sqlalchemy import (
    MetaData,
    Table,
    Column,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.engine import Engine, Result
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from sqlalchemy.types import NullType

from langchain_core.embeddings import Embeddings

from .utils import truncate_word, _safe_filename


class SQLiteDatabase:
    """SQLAlchemy wrapper around a SQLite database with column comments support."""

    def _render_type(self, col_type: Any, *, default: str = "TEXT") -> str:
        """Render SQLAlchemy type using this engine's dialect when possible."""
        if col_type is None or isinstance(col_type, NullType):
            return default
        try:
            compiled = col_type.compile(dialect=self._engine.dialect)
            if isinstance(compiled, str) and compiled.strip():
                return compiled.strip()
        except Exception:
            pass
        try:
            rendered = str(col_type)
            return rendered.strip() if rendered.strip() else default
        except Exception:
            return default

    def __init__(
        self,
        engine: Engine,
        ignore_tables: Optional[List[str]] = None,
        include_tables: Optional[List[str]] = None,
        indexes_in_table_info: bool = False,
        max_string_length: int = 200,
        lazy_table_reflection: bool = False,
        faiss_dir: Optional[str] = None,
        embeddings: Optional[Embeddings] = None,
    ):
        """
        Create SQLite database wrapper.
        
        Args:
            engine: SQLAlchemy engine connected to SQLite database
            ignore_tables: List of table names to ignore
            include_tables: List of table names to include (mutually exclusive with ignore_tables)
            indexes_in_table_info: Whether to include index information in table info
            max_string_length: Maximum string length for truncating values
            lazy_table_reflection: Whether to lazily reflect tables
            faiss_dir: Root directory that stores FAISS artifacts (see create_faiss)
            embeddings: Optional pre-initialized InfinityEmbeddings instance to reuse
            embed_model: Model name for InfinityEmbeddings (used when embeddings is None)
            infinity_api_url: Infinity API endpoint (used when embeddings is None)
        """
        self._engine = engine
        if self._engine.dialect.name != "sqlite":
            raise ValueError("SQLiteDatabase only supports SQLite databases")
        
        if include_tables and ignore_tables:
            raise ValueError("Cannot specify both include_tables and ignore_tables")

        self._inspector = inspect(self._engine)
        self._all_tables = set(self._inspector.get_table_names())

        self._include_tables = set(include_tables) if include_tables else set()
        if self._include_tables:
            missing_tables = self._include_tables - self._all_tables
            if missing_tables:
                raise ValueError(f"include_tables {missing_tables} not found in database")
        
        self._ignore_tables = set(ignore_tables) if ignore_tables else set()
        if self._ignore_tables:
            missing_tables = self._ignore_tables - self._all_tables
            if missing_tables:
                raise ValueError(f"ignore_tables {missing_tables} not found in database")
        
        usable_tables = self.get_usable_table_names()
        self._usable_tables = set(usable_tables) if usable_tables else self._all_tables

        self._indexes_in_table_info = indexes_in_table_info
        self._max_string_length = max_string_length
        self._faiss_dir = faiss_dir
        self._faiss_indexes: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._faiss_embeddings = embeddings

        self._metadata = MetaData()
        if not lazy_table_reflection:
            self._metadata.reflect(
                bind=self._engine,
                only=list(self._usable_tables),
            )

        if self._faiss_dir:
            self._load_faiss_indexes()


    @classmethod
    def from_uri(
        cls,
        database_uri: str,
        engine_args: Optional[dict] = None,
        **kwargs: Any,
    ) -> "SQLiteDatabase":
        """Construct a SQLiteDatabase from URI."""
        _engine_args = engine_args or {}
        return cls(create_engine(database_uri, **_engine_args), **kwargs)


    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        return "SQLite"


    def get_usable_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        if self._include_tables:
            base = set(self._include_tables)
        else:
            base = self._all_tables - self._ignore_tables

        # filter out metadata tables (companion EAV tables)
        base = {tbl for tbl in base if not tbl.endswith("__metadata")}
        return sorted(base)


    def get_column_datatype(
        self,
        table_name: str,
        column_name: str,
        default: str = "TEXT",
    ) -> str:
        """
        Return SQL datatype for a column in a table.

        Notes:
        - Uses SQLAlchemy inspector, so it does not require table reflection.
        - Returns `default` when the table/column is not found or the type is unknown.
        """
        all_table_names = set(self.get_usable_table_names())
        if table_name not in all_table_names:
            raise ValueError(
                f"Table '{table_name}' not found in database. Available tables: {sorted(all_table_names)}"
            )

        try:
            cols = self._inspector.get_columns(table_name)
        except SQLAlchemyError:
            return default

        for col in cols:
            if col.get("name") != column_name:
                continue
            col_type = col.get("type")
            return self._render_type(col_type, default=default)

        return default


    def get_table_info(
        self,
        table_name: str,
        get_col_comments: bool = False,
        allowed_col_names: Optional[List[str]] = None,
        sample_count: Optional[int] = None,
    ) -> str:
        """
        Get information about a specified table.

        Args:
            table_name: Name of the table to get info for
            get_col_comments: Whether to include column comments in the output
            allowed_col_names: If provided, only include these columns in the output.
                              If None, include all columns.
            sample_count: Number of distinct example values to include for each column.
                          If None, no example values are included.

        Returns:
            String containing table schema (CREATE TABLE statement) and optionally
            column comments and sample rows.
        """
        all_table_names = list(self.get_usable_table_names())
        if table_name not in all_table_names:
            raise ValueError(f"Table '{table_name}' not found in database. Available tables: {all_table_names}")

        # Ensure table is reflected
        metadata_table_names = [tbl.name for tbl in self._metadata.sorted_tables]
        if table_name not in metadata_table_names:
            self._metadata.reflect(
                bind=self._engine,
                only=[table_name],
            )

        # Find the table object
        table = None
        for tbl in self._metadata.sorted_tables:
            if tbl.name == table_name:
                table = tbl
                break

        if table is None:
            raise ValueError(f"Table '{table_name}' could not be reflected")

        # Remove NullType columns
        try:
            for _, v in table.columns.items():
                if type(v.type) is NullType:
                    table._columns.remove(v)
        except AttributeError:
            for _, v in dict(table.columns).items():
                if type(v.type) is NullType:
                    table._columns.remove(v)

        # Filter columns if allowed_col_names is specified
        display_columns = list(table.columns) if not allowed_col_names else [col for col in table.columns if col.name in allowed_col_names]
        if not display_columns:
            raise ValueError(f"No matching columns found. Requested: {allowed_col_names}")

        # Get sample values for columns if requested
        column_sample_values: Dict[str, List[str]] = {}
        if sample_count:
            column_sample_values = self._get_sample_values(
                table, display_columns, sample_count
            )

        # Build custom CREATE TABLE statement with filtered columns
        col_defs = []
        column_descriptions = (
            self._get_column_descriptions_from_metadata(table_name)
            if get_col_comments
            else {}
        )
        for col in display_columns:
            col_type = self._render_type(col.type, default="TEXT")
            col_def = f'\t"{col.name}" {col_type}'
            
            # Build comment with description and example values
            comment_parts = []
            col_cmt = column_descriptions.get(col.name, "")
            if col_cmt:
                comment_parts.append(col_cmt)
            
            # Add sample values if available
            if col.name in column_sample_values and column_sample_values[col.name]:
                sample_values = column_sample_values[col.name]
                examples_str = ", ".join(str(v) for v in sample_values)
                comment_parts.append(f"Ví dụ: {examples_str},...")
            
            if comment_parts:
                comment_text = " ".join(comment_parts)
                col_def = f"{col_def}\t/* {comment_text} */"
            
            col_defs.append(col_def)

        col_defs.sort()        
        create_table = f'CREATE TABLE "{table_name}" (\n' + ", \n".join(col_defs) + "\n)"

        table_info = f"{create_table.rstrip()}"
            
        # Add indexes if needed
        if self._indexes_in_table_info:
            table_info += "\n\n/*"
            table_info += f"\n{self._get_table_indexes(table)}\n"
            table_info += "*/"

        return table_info


    def _get_column_descriptions_from_metadata(
        self, table_name: str
    ) -> Dict[str, str]:
        """
        Fetch column descriptions from the metadata EAV table created alongside the data table.

        Expects a companion table named "{table_name}__metadata" with rows:
            entity = column name
            attribute = "description"
            value = description text
        """
        metadata_table = f"{table_name}__metadata"
        if metadata_table not in self._all_tables:
            return {}

        try:
            query = text(
                f'SELECT entity, value FROM "{metadata_table}" WHERE attribute = :attr'
            )
            with self._engine.connect() as connection:
                result: Result = connection.execute(query, {"attr": "description"})
                return {row[0]: row[1] for row in result if row[1] is not None}
        except (ProgrammingError, SQLAlchemyError):
            return {}


    def get_column_groups(self, table_name: str) -> List[List[str]]:
        """
        Return column groups for a table based on its metadata companion table.

        Reads rows where attribute == "group" from "{table_name}__metadata" and
        builds a list of column-name lists, ordered by group id.
        """
        metadata_table = f"{table_name}__metadata"
        if metadata_table not in self._all_tables:
            return []

        groups: Dict[int, List[str]] = {}
        try:
            query = text(
                f'SELECT entity, value FROM "{metadata_table}" WHERE attribute = :attr'
            )
            with self._engine.connect() as connection:
                result: Result = connection.execute(query, {"attr": "group"})
                for entity, value in result:
                    if value is None:
                        continue
                    try:
                        group_id = int(value)
                    except (TypeError, ValueError):
                        continue
                    groups.setdefault(group_id, []).append(entity)
        except (ProgrammingError, SQLAlchemyError):
            return []

        if not groups:
            return []

        return [groups[idx] for idx in sorted(groups.keys())]


    def _get_table_indexes(self, table: Table) -> str:
        """Get formatted index information for a table."""
        indexes = self._inspector.get_indexes(table.name)
        indexes_formatted = "\n".join(
            f"Name: {idx['name']}, Unique: {idx['unique']}, Columns: {idx['column_names']}"
            for idx in indexes
        )
        return f"Table Indexes:\n{indexes_formatted}"


    def _get_sample_values(
        self,
        table: Table,
        columns: List[Column],
        sample_count: int,
    ) -> Dict[str, List[str]]:
        """
        Get up to sample_count distinct example values per column.

        Strings are quoted to reflect their type. Values longer than 100 chars are skipped.
        """
        if sample_count <= 0:
            return {}

        column_sample_values: Dict[str, List[str]] = {col.name: [] for col in columns}
        for col in columns:
            query = text(
                f'SELECT DISTINCT "{col.name}" '
                f'FROM "{table.name}" '
                f'WHERE "{col.name}" IS NOT NULL '
                f"LIMIT {sample_count}"
            )

            try:
                with self._engine.connect() as connection:
                    result = connection.execute(query)
                    remaining_length = 1000
                    for val, in result:
                        val_str = str(val)
                        # Represent type: quote strings, leave others as-is
                        display_val = f'"{val_str}"' if isinstance(val, str) else val_str
                        column_sample_values[col.name].append(display_val)
                        remaining_length -= len(display_val)
                        if remaining_length <= 0:
                            break

            except ProgrammingError:
                continue

        return column_sample_values


    def _execute(
        self,
        command: str,
        fetch: Literal["all", "one", "cursor"] = "all",
        *,
        parameters: Optional[Dict[str, Any]] = None,
        execution_options: Optional[Dict[str, Any]] = None,
    ) -> Union[Sequence[Dict[str, Any]], Result]:
        """Execute SQL command through underlying engine."""
        parameters = parameters or {}
        execution_options = execution_options or {}
        
        with self._engine.begin() as connection:
            cursor = connection.execute(
                text(command),
                parameters,
                execution_options=execution_options,
            )

            if cursor.returns_rows:
                if fetch == "all":
                    result = [x._asdict() for x in cursor.fetchall()]
                elif fetch == "one":
                    first_result = cursor.fetchone()
                    result = [] if first_result is None else [first_result._asdict()]
                elif fetch == "cursor":
                    return cursor
                else:
                    raise ValueError("Fetch parameter must be either 'one', 'all', or 'cursor'")
                return result
        return []


    def run(
        self,
        command: str,
        fetch: Literal["all", "one", "cursor"] = "all",
        include_columns: bool = False,
        *,
        parameters: Optional[Dict[str, Any]] = None,
        execution_options: Optional[Dict[str, Any]] = None,
    ) -> Union[Sequence[Dict[str, Any]], Sequence[Tuple[Any, ...]], Result[Any]]:
        """Execute a SQL command and return a string representing the results."""
        result = self._execute(
            command, fetch, parameters=parameters, execution_options=execution_options
        )

        if fetch == "cursor":
            return result

        if include_columns:
            return [
                {
                    column: truncate_word(value, length=self._max_string_length)
                    for column, value in r.items()
                }
                for r in result
            ]
        else:
            return [
                tuple(
                    truncate_word(value, length=self._max_string_length)
                    for value in r.values()
                )
                for r in result
            ]


    def run_no_throw(
        self,
        command: str,
        fetch: Literal["all", "one"] = "all",
        include_columns: bool = False,
        *,
        parameters: Optional[Dict[str, Any]] = None,
        execution_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a SQL command and return results or error message."""
        try:
            res = self.run(
                command,
                fetch,
                parameters=parameters,
                execution_options=execution_options,
                include_columns=include_columns,
            )
            return {
                "result": res,
                "error": None,
            }
        except SQLAlchemyError as e:
            return {
                "result": [],
                "error": f"Error: {e}",
            }


    def get_table_info_no_throw(
        self,
        table_name: str,
        get_col_comments: bool = False,
        allowed_col_names: Optional[List[str]] = None,
        sample_count: Optional[int] = None,
    ) -> str:
        """Get table info without throwing exceptions."""
        try:
            return self.get_table_info(
                table_name,
                get_col_comments=get_col_comments,
                allowed_col_names=allowed_col_names,
                sample_count=sample_count,
            )
        except ValueError as e:
            return f"Error: {e}"


    def get_context(self) -> Dict[str, Any]:
        """Return db context that you may want in agent prompt."""
        table_names = list(self.get_usable_table_names())
        # Get info for all tables
        table_infos = []
        for tbl in table_names:
            table_infos.append(self.get_table_info_no_throw(tbl))
        table_info = "\n\n".join(table_infos)
        return {"table_info": table_info, "table_names": ", ".join(table_names)}


    def _load_faiss_indexes(self) -> None:
        """Eagerly load FAISS indexes and their value mappings if the directory is present."""
        if not self._faiss_dir:
            return

        for table_name in self._usable_tables:
            table_dir = os.path.join(
                self._faiss_dir,
                _safe_filename(table_name),
            )
            if not os.path.isdir(table_dir):
                continue

            # Build a lookup of column names to quickly check candidate files.
            try:
                columns = {c["name"] for c in self._inspector.get_columns(table_name)}
            except SQLAlchemyError:
                continue

            for col_name in columns:
                index_path = os.path.join(
                    table_dir, f"{_safe_filename(col_name)}.faiss"
                )
                values_path = os.path.join(
                    table_dir, f"{_safe_filename(col_name)}.json"
                )
                if not (os.path.exists(index_path) and os.path.exists(values_path)):
                    continue

                try:
                    index = faiss.read_index(index_path)
                    with open(values_path, "r", encoding="utf-8") as f:
                        values = json.load(f) or []
                    if not isinstance(values, list):
                        continue

                    metric = getattr(index, "metric_type", None)
                    normalize = metric == faiss.METRIC_INNER_PRODUCT

                    # Guard against mismatched artifacts.
                    if hasattr(index, "ntotal") and index.ntotal != len(values):
                        # Skip inconsistent artifacts to avoid incorrect lookups.
                        continue

                    self._faiss_indexes.setdefault(table_name, {})[col_name] = {
                        "index": index,
                        "values": values,
                        "normalize": normalize,
                    }
                except Exception:
                    # Ignore malformed artifacts; consumers can still use SQL methods.
                    continue


    async def batch_search_similar_values(
        self,
        batch_cells: List[Tuple[str, str, str]],
        k: int = 5,
        max_concurrency: int = 10,
    ) -> List[List[Dict[str, Any]]]:
        """
        Asynchronously search for similar values for a batch of cells.
        
        Args:
            batch_cells: List of (table_name, column_name, cell_value) tuples.
            k: Number of nearest neighbors to retrieve.
            max_concurrency: Maximum number of concurrent threads for FAISS searching.

        Returns:
            A list of result lists corresponding to the order of input batch_cells.
            If an index is missing for a cell, returns an empty list for that slot.
        """
        if not self._faiss_indexes:
            raise ValueError("FAISS indexes are not loaded for this database")
        if self._faiss_embeddings is None:
            raise ValueError("Embeddings client is not configured")

        # 1. Validation and preparation
        valid_queries = []
        valid_indices = []
        texts_to_embed = []
        
        # Initialize results with empty lists (preserving order)
        results: List[List[Dict[str, Any]]] = [[] for _ in batch_cells]

        for i, (table, col, val) in enumerate(batch_cells):
            table_indexes = self._faiss_indexes.get(table)
            if table_indexes and col in table_indexes:
                valid_queries.append((table, col))
                valid_indices.append(i)
                texts_to_embed.append(str(val))
            # Note: Invalid cells (no index found) remain [] in the results list

        if not valid_queries:
            return results

        # 2. Batch Embedding (I/O Bound)
        # Use the async batch embedding method from LangChain
        embeddings = await self._get_batch_embeddings(texts_to_embed)

        # 3. Parallel FAISS Search (CPU Bound, offloaded to threads)
        semaphore = asyncio.Semaphore(max_concurrency)
        tasks = []

        for i, embedding in zip(valid_indices, embeddings):
            table_name, column_name = batch_cells[i][0], batch_cells[i][1]
            index_data = self._faiss_indexes[table_name][column_name]
            
            task = self._execute_search_with_semaphore(
                semaphore=semaphore,
                index_data=index_data,
                vector=embedding,
                k=k
            )
            tasks.append(task)

        # Wait for all search tasks to complete
        search_results = await asyncio.gather(*tasks)

        # 4. Map results back to original positions
        for original_idx, res in zip(valid_indices, search_results):
            results[original_idx] = res

        return results


    async def _get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Helper to fetch embeddings asynchronously."""
        if self._faiss_embeddings is None:
            raise ValueError("Infinity embeddings client is not configured")
        if not hasattr(self._faiss_embeddings, "aembed_documents"):
            raise ValueError("Infinity embeddings client does not support aembed_documents")
        return await self._faiss_embeddings.aembed_documents(texts)


    async def _execute_search_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        index_data: Dict[str, Any],
        vector: List[float],
        k: int
    ) -> List[Dict[str, Any]]:
        """Acquires semaphore and offloads CPU work to a thread."""
        await semaphore.acquire()
        try:
            return await asyncio.to_thread(
                self._run_faiss_search_job,
                index=index_data["index"],
                values=index_data["values"],
                normalize=index_data["normalize"],
                vector=vector,
                k=k
            )
        except Exception as e:
            raise e
        finally:
            semaphore.release()


    @staticmethod
    def _run_faiss_search_job(
        index: Any,
        values: List[str],
        normalize: bool,
        vector: List[float],
        k: int
    ) -> List[Dict[str, Any]]:
        """
        Pure CPU-bound static method to perform the FAISS search.
        Running this in a separate thread avoids blocking the asyncio event loop.
        """
        x = np.asarray(vector, dtype="float32")
        
        # Ensure correct shape (1, embedding_dim)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if normalize:
            faiss.normalize_L2(x)

        distances, indices = index.search(x, min(k, len(values)))
        
        results: List[Dict[str, Any]] = []
        found_indices = indices[0]
        found_distances = distances[0]

        for idx, score in zip(found_indices, found_distances):
            if idx < 0 or idx >= len(values):
                continue
            results.append({
                "value": values[int(idx)],
                "score": float(score)
            })
            
        return results