from typing import Any, Dict, Iterable, List, Literal, Sequence, Tuple, Union, Optional
from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.engine import Engine, Result
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from sqlalchemy.types import NullType

from .utils import truncate_word


class SQLiteDatabase:
    """SQLAlchemy wrapper around a SQLite database with column comments support."""

    def __init__(
        self,
        engine: Engine,
        ignore_tables: Optional[List[str]] = None,
        include_tables: Optional[List[str]] = None,
        indexes_in_table_info: bool = False,
        max_string_length: int = 300,
        lazy_table_reflection: bool = False,
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

        self._metadata = MetaData()
        if not lazy_table_reflection:
            self._metadata.reflect(
                bind=self._engine,
                only=list(self._usable_tables),
            )


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
        return "sqlite"


    def get_usable_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        if self._include_tables:
            return sorted(self._include_tables)
        return sorted(self._all_tables - self._ignore_tables)


    def get_table_info(
        self,
        table_name: str,
        get_col_comments: bool = False,
        allowed_col_names: Optional[List[str]] = None,
        sample_row_ids: Optional[List[int]] = None,
    ) -> str:
        """
        Get information about a specified table.

        Args:
            table_name: Name of the table to get info for
            get_col_comments: Whether to include column comments in the output
            allowed_col_names: If provided, only include these columns in the output.
                              If None, include all columns.
            sample_row_ids: List of row IDs to sample. If provided, sample rows with
                           these IDs using ROWID. If None, no sample rows are included.

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

        # Build custom CREATE TABLE statement with filtered columns
        col_defs = []
        column_descriptions = (
            self._get_column_descriptions_from_metadata(table_name)
            if get_col_comments
            else {}
        )
        for col in display_columns:
            col_type = str(col.type) if not isinstance(col.type, NullType) else "TEXT"
            col_def = f'\t"{col.name}" {col_type}'
            col_cmt = column_descriptions.get(col.name, "")
            if col_cmt:
                col_def = f"{col_def}\t/* {col_cmt} */"
            col_defs.append(col_def)

        col_defs.sort()        
        create_table = f'CREATE TABLE "{table_name}" (\n' + ", \n".join(col_defs) + "\n)"

        table_info = f"{create_table.rstrip()}"
            
        # Add indexes and sample rows
        has_extra_info = self._indexes_in_table_info or sample_row_ids
        if has_extra_info:
            table_info += "\n\n/*"
        
        if self._indexes_in_table_info:
            table_info += f"\n{self._get_table_indexes(table)}\n"
        
        if sample_row_ids:
            table_info += f"\n{self._get_sample_rows_by_ids(table, sample_row_ids, allowed_col_names)}\n"
        
        if has_extra_info:
            table_info += "*/"

        return table_info

    def _get_column_descriptions_from_metadata(self, table_name: str) -> Dict[str, str]:
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


    def _get_table_indexes(self, table: Table) -> str:
        """Get formatted index information for a table."""
        indexes = self._inspector.get_indexes(table.name)
        indexes_formatted = "\n".join(
            f"Name: {idx['name']}, Unique: {idx['unique']}, Columns: {idx['column_names']}"
            for idx in indexes
        )
        return f"Table Indexes:\n{indexes_formatted}"


    def _get_sample_rows_by_ids(
        self,
        table: Table,
        row_ids: List[int],
        allowed_col_names: Optional[List[str]] = None,
    ) -> str:
        """
        Get sample rows by their ROWID.

        Args:
            table: SQLAlchemy Table object
            row_ids: List of ROWIDs to fetch
            allowed_col_names: If provided, only include these columns

        Returns:
            Formatted string with sample rows
        """
        if allowed_col_names:
            columns = [col for col in table.columns if col.name in allowed_col_names]
        else:
            columns = list(table.columns)

        columns_str = "\t".join([col.name for col in columns])

        try:
            # Build query with ROWID filter
            col_names = [f'"{col.name}"' for col in columns]
            col_list = ", ".join(col_names)
            id_list = ", ".join(str(rid) for rid in row_ids)
            query = f'SELECT {col_list} FROM "{table.name}" WHERE ROWID IN ({id_list})'

            with self._engine.connect() as connection:
                result = connection.execute(text(query))
                sample_rows = [
                    [str(val)[:100] for val in row]
                    for row in result
                ]

            sample_rows_str = "\n".join(["\t".join(row) for row in sample_rows])

        except ProgrammingError:
            sample_rows_str = ""

        return (
            f"{len(row_ids)} sample rows from {table.name} table (by ROWID):\n"
            f"{columns_str}\n"
            f"{sample_rows_str}"
        )


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
        sample_row_ids: Optional[List[int]] = None,
    ) -> str:
        """Get table info without throwing exceptions."""
        try:
            return self.get_table_info(
                table_name,
                get_col_comments=get_col_comments,
                allowed_col_names=allowed_col_names,
                sample_row_ids=sample_row_ids,
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