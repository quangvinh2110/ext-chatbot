import copy
import asyncio
from typing import Optional, Dict, Any, Tuple, List

from ...parser.excel_parser import TableInfo
from ...utils import set_seed
from .header_footer_identifier import HeaderFooterIdentifier
from .header_generator import HeaderGenerator
from .column_grouper import ColumnGrouper
from .schema_designer import SchemaDesigner
from .data_transformer import DataTransformer
from .utils import format_header_rows
set_seed(42)



class TableProcessor:
    """
    Processes tables extracted by ExcelParser using an LLM to determine structure and schema.
    """
    
    def __init__(
        self, 
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the TableProcessor.
        
        Args:
            base_url: Optional base URL. If not provided, uses LLM_BASE_URL env var.
            api_key: Optional API key. If not provided, uses LLM_API_KEY env var.
            model: Optional model name. If not provided, uses LLM_MODEL env var.
            max_concurrent_requests: Maximum number of concurrent requests. Defaults to 10.
        """
        self.header_footer_identifier = HeaderFooterIdentifier(
            base_url=base_url,
            api_key=api_key,
            model=model,
        )
        self.header_generator = HeaderGenerator(
            base_url=base_url,
            api_key=api_key,
            model=model,
        )
        self.column_grouper = ColumnGrouper(
            base_url=base_url,
            api_key=api_key,
            model=model,
        )
        self.schema_designer = SchemaDesigner(
            base_url=base_url,
            api_key=api_key,
            model=model,
        )
        self.data_transformer = DataTransformer(
            base_url=base_url,
            api_key=api_key,
            model=model,
        )

    def __call__(
        self,
        table: TableInfo,
        max_retries: int = 5,
        max_concurrent_requests: int = 10,
    ) -> Dict[str, Any]:
        """
        Process a single table to extract structure and schema information.
        
        Args:
            table: TableInfo object from ExcelParser.
            max_retries (int, optional): Maximum number of retries for API call and JSON parsing. Defaults to 5.
        Returns:
            Dict containing 'structure_info' and 'transformed_data'.
        """
        table_rows = table.data.tolist()
        table_rows = self.preprocess_data(table_rows)

        print("Identifying header and footer...")
        header_footer_info = self.header_footer_identifier(
            table_rows, 
            sheet_name=table.sheet_name,
            max_retries=max_retries
        )
        print(header_footer_info)

        header_footer_info["generated_header"] = None
        header_indices = header_footer_info["header_indices"] or []
        footer_indices = header_footer_info.get("footer_indices") or []

        if not header_indices:
            print("Generating header...")
            generated_header = self.header_generator(
                table_rows, 
                sheet_name=table.sheet_name,
                max_retries=max_retries
            )
            print(generated_header)
            header_footer_info["generated_header"] = generated_header
            formatted_header = generated_header
        else:
            header_rows = [table_rows[i] for i in header_indices]
            formatted_header = format_header_rows(header_rows)


        data_rows = []
        for i, row in enumerate(table_rows):
            if i not in header_indices and i not in footer_indices:
                data_rows.append(row)
        
        if not data_rows:
            raise ValueError("No rows to process. Please re-check the header and footer indices.")

        print("Grouping columns...")
        column_groups = self.column_grouper(
            data_rows=data_rows,
            formatted_header=formatted_header,
            sheet_name=table.sheet_name,
            method="hybrid",
            max_retries=max_retries,
        )
        col_to_data_map: Dict[str, Tuple[Any]] = {
            col: data 
            for col, data in zip(formatted_header, zip(*data_rows))
        }

        data_groups = [
            [
                list(row)
                for row in zip(*[col_to_data_map[col] for col in group])
            ]
            for group in column_groups
        ]
        print("Designing schema...")
        pydantic_schema = asyncio.run(self.schema_designer(
            data_groups=data_groups, 
            column_groups=column_groups, 
            sheet_name=table.sheet_name,
            max_retries=max_retries,
            without_new_cols=(sum(len(group) for group in column_groups) > 20),
        ))
        print(pydantic_schema)

        new_column_groups = []
        for group in column_groups:
            new_group = copy.deepcopy(group)
            for new_col in pydantic_schema["properties"]:
                if new_col not in new_group:
                    for col in group:
                        if col.replace(" ", "_") in new_col.replace(" ", "_"):
                            new_group.append(new_col)
                            break
            new_group.sort()
            new_column_groups.append(new_group)

        print("Transforming data...")
        transformed_data = asyncio.run(self.data_transformer(
            data_rows=data_rows, 
            formatted_header=formatted_header, 
            pydantic_schema=pydantic_schema, 
            max_retries=max_retries,
            max_concurrent_requests=max_concurrent_requests,
        ))

        return {
            "header_footer_info": header_footer_info,
            "pydantic_schema": pydantic_schema,
            "column_groups": new_column_groups,
            "transformed_data": transformed_data
        }


    def preprocess_data(self, table_rows: List[List[Any]]) -> List[List[Any]]:
        print("Preprocessing data...")
        preprocessed_rows = []
        for row in table_rows:
            new_row = []
            for cell in row:
                if isinstance(cell, str):
                    new_row.append(cell.strip())
                else:
                    new_row.append(cell)
            preprocessed_rows.append(new_row)
        return preprocessed_rows


# In design_schema method, instead of generating pydantic schema, generate SQL schema for different dialects like sqlite, mysql, postgresql, etc.
# In transform_data method, instead of generating a json object, generate a SQL query that can be used to insert the data into the database, then run the query to check if the data is inserted correctly.