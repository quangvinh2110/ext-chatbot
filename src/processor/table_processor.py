import re
import json
import os
import random
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from openai import OpenAI, AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from ..parser.excel_parser import TableInfo
from ..prompts import (
    TRANSFORM_DATA_TEMPLATE,
    IDENTIFY_HEADER_FOOTER_TEMPLATE,
    DESIGN_SCHEMA_TEMPLATE,
    GENERATE_HEADER_TEMPLATE,
    GROUP_COLUMNS_TEMPLATE
)
from ..utils import set_seed


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
        max_concurrent_requests: int = 10,
        max_retries: int = 3,
    ):
        """
        Initialize the TableProcessor.
        
        Args:
            base_url: Optional base URL. If not provided, uses LLM_BASE_URL env var.
            api_key: Optional API key. If not provided, uses LLM_API_KEY env var.
            model: Optional model name. If not provided, uses LLM_MODEL env var.
            max_concurrent_requests: Maximum number of concurrent requests. Defaults to 10.
        """
        self.max_concurrent_requests = max_concurrent_requests
        self.client = OpenAI(
            base_url=base_url or os.getenv("LLM_BASE_URL"),
            api_key=api_key or os.getenv("LLM_API_KEY")
        )
        
        self.async_client = AsyncOpenAI(
            base_url=base_url or os.getenv("LLM_BASE_URL"),
            api_key=api_key or os.getenv("LLM_API_KEY")
        )
        
        model_name = model or os.getenv("LLM_MODEL")
        if not model_name:
            raise ValueError("LLM_MODEL not set. Please provide a model name or set LLM_MODEL environment variable.")
        self.model = model_name
        
        self.json_pattern = re.compile(r"```json\n(.*?)\n```", re.DOTALL)

    def __call__(
        self,
        table: TableInfo,
        max_retries: int = 5,
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
        print("Identifying header and footer...")
        header_footer_info = self.identify_header_footer(table_rows, max_retries)
        print(header_footer_info)

        header_footer_info["generated_header"] = None
        header_indices = header_footer_info["header_indices"] or []
        footer_indices = header_footer_info.get("footer_indices") or []

        if not header_indices:
            print("Generating header...")
            generated_header = self.generate_header(table_rows, max_retries)
            print(generated_header)
            header_footer_info["generated_header"] = generated_header
            formatted_header = generated_header
        else:
            header_rows = [table_rows[i] for i in header_indices]
            formatted_header = self._format_header_rows(header_rows)


        data_rows = []
        for i, row in enumerate(table_rows):
            if i not in header_indices and i not in footer_indices:
                data_rows.append(row)
        
        if not data_rows:
            raise ValueError("No rows to process. Please re-check the header and footer indices.")

        print("Grouping columns...")
        column_groups = self.group_columns(
            data_rows=data_rows,
            formatted_header=formatted_header,
            max_retries=max_retries
        )
        print(column_groups)

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
        pydantic_schema = asyncio.run(self.design_schema(
            data_groups=data_groups, 
            column_groups=column_groups, 
            max_retries=max_retries
        ))
        print(pydantic_schema)

        print("Transforming data...")
        transformed_data = asyncio.run(self.transform_data(
            data_rows=data_rows, 
            formatted_header=formatted_header, 
            pydantic_schema=pydantic_schema, 
            max_retries=max_retries
        ))

        return {
            "header_footer_info": header_footer_info,
            "pydantic_schema": pydantic_schema,
            "transformed_data": transformed_data
        }


    def identify_header_footer(
        self,
        table_rows: List[List[Any]],
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        snippet = self._format_table_data_snippet(table_rows)
        prompt = IDENTIFY_HEADER_FOOTER_TEMPLATE.replace(
            "{{table_data_snippet}}",
            snippet
        )
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    top_p=0.8,
                    presence_penalty=1,
                    extra_body = {
                        "chat_template_kwargs": {'enable_thinking': False},
                        "top_k": 20,
                        "mip_p": 0,
                    },
                    timeout=20,
                )
                
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("LLM returned empty content")
                
                header_footer_info = self._extract_json(content)
                if "header_indices" not in header_footer_info or "footer_indices" not in header_footer_info:
                    raise ValueError("Failed to identify header and footer")
                    
                return {
                    "header_indices": header_footer_info["header_indices"],
                    "footer_indices": header_footer_info["footer_indices"],
                }
            
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                continue

        return {"header_indices": None, "footer_indices": None}
        

    def generate_header(
        self,
        table_rows: List[List[Any]],
        max_retries: int = 3,
    ) -> List[str]:
        snippet = self._format_table_data_snippet_with_header(
            formatted_header=[f"col{i}" for i in range(len(table_rows[0]))],
            data_rows=table_rows,
        )
        prompt = GENERATE_HEADER_TEMPLATE.replace(
            "{{table_data_snippet}}", snippet
        )
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    top_p=0.8,
                    presence_penalty=1,
                    extra_body = {
                        "chat_template_kwargs": {'enable_thinking': False},
                        "top_k": 20,
                        "mip_p": 0,
                    },
                    timeout=10,
                )
                
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("LLM returned empty content")
                generated_header = self._extract_json(content)
                if "generated_header" not in generated_header or not generated_header["generated_header"]:
                    raise ValueError("Generated header not found in response")
                elif len(generated_header["generated_header"]) != len(table_rows[0]):
                    raise ValueError("Generated header does not match the number of columns in the table")
                return generated_header["generated_header"]
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                continue
                
        raise ValueError("Failed to process request after max retries")


    async def design_schema(
        self,
        data_groups: List[List[List[Any]]],
        column_groups: List[List[str]],
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Design schema for each data group asynchronously.
        """

        queue: List[Tuple[List[List[Any]], List[str]]] = [
            (data_group, column_group) 
            for data_group, column_group in zip(data_groups, column_groups)
        ]
        results = []
        for attempt in range(max_retries):
            if not queue:
                break

            print(f"Attempt {attempt + 1}/{max_retries}: Processing {len(queue)} groups...")
            
            # Process batch
            batch_results = await self._design_schema_for_groups(queue)

            # Collect successes and prepare retries
            successful_results = []
            failed_items = []
            
            for res in batch_results:
                if res["success"]:
                    successful_results.append(res)
                else:
                    failed_items.append(res["input_item"])
                    
            results.extend(successful_results)
            queue = failed_items

            if attempt == max_retries - 1 and queue:
                errors = [res["error"] for res in batch_results if not res["success"]]
                raise ValueError(
                    "Failed to design schema for all groups after max retries"
                    f", Errors from last attempt: {errors}"          
                )

        # Step 3: Merge all schemas
        print("Merging schemas...")
        merged_schema = self._merge_schemas([
            result["pydantic_schema"] for result in results
        ])
        
        return merged_schema
    
    
    def group_columns(
        self,
        data_rows: List[List[Any]],
        formatted_header: List[str],
        max_retries: int = 3,
    ) -> List[List[str]]:
        """
        Group columns by their semantic meaning using LLM.
        
        Args:
            data_rows: List of data rows
            formatted_header: List of column names
            max_retries: Maximum number of retries
            
        Returns:
            List of groups, where each group is a list of column names
        """
        if len(formatted_header) <= 8:
            return [formatted_header]

        num_samples = min(5, len(data_rows))
        sample_rows = random.sample(data_rows, num_samples)
        snippet = self._format_table_data_snippet_with_header(
            formatted_header=formatted_header,
            data_rows=sample_rows,
        )
        
        prompt = GROUP_COLUMNS_TEMPLATE.replace(
            "{{table_data_snippet}}", snippet
        )
        
        column_groups: List[List[str]] = []
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    top_p=0.8,
                    presence_penalty=1,
                    extra_body = {
                        "chat_template_kwargs": {'enable_thinking': False},
                        "top_k": 20,
                        "mip_p": 0,
                    },
                )
                
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("LLM returned empty content")

                result = self._extract_json(content)
                
                # Validate the result
                if not result.get("column_groups"):
                    raise ValueError("Failed to group columns")
                
                column_groups = [group["columns"] for group in result["column_groups"]]
                
                # Validate that all columns are included and no duplicates
                all_columns_in_groups = []
                for group in column_groups:
                    all_columns_in_groups.extend(group)
                
                if set(all_columns_in_groups) != set(formatted_header):
                    raise ValueError("Column groups do not match the original header")
                
                if len(all_columns_in_groups) != len(set(all_columns_in_groups)):
                    raise ValueError("Duplicate columns found in groups")
                
                return column_groups
            
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                continue
        
        # Fallback: treat each column as its own group
        print("Failed to group columns, falling back to the last result")
        fixed_column_groups = []
        all_columns_in_groups = []
        for column_group in column_groups:
            fixed_column_group = []
            for column in column_group:
                if column in formatted_header:
                    fixed_column_group.append(column)
                    all_columns_in_groups.append(column)
            fixed_column_groups.append(fixed_column_group)
        for column in formatted_header:
            if column not in all_columns_in_groups:
                fixed_column_groups.append([column])
        return fixed_column_groups
    
    
    async def _design_schema_for_groups(
        self,
        queue: List[Tuple[List[List[Any]], List[str]]],
    ) -> List[Dict[str, Any]]:
        """
        Design schema for each column group asynchronously.
        
        Args:
            queue: List of data groups and column groups
            
        Returns:
            List of schemas, one for each group
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        tasks = [
            self._design_schema_for_single_group(
                data_group=data_group,
                column_group=column_group,
                semaphore=semaphore
            )
            for data_group, column_group in queue
        ]
        return await tqdm_asyncio.gather(*tasks, desc="Designing schema for groups")
    
    
    async def _design_schema_for_single_group(
        self,
        data_group: List[List[Any]],
        column_group: List[str],
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        """
        Design schema for a single column group.
        
        Args:
            data_group: List of data rows
            column_group: List of all column names
            semaphore: Semaphore for rate limiting
            
        Returns:
            Schema for this group
        """
        await semaphore.acquire()

        snippet = self._format_table_data_snippet_with_header(
            formatted_header=column_group,
            data_rows=data_group,
            sample_size=10,
        )
        prompt = DESIGN_SCHEMA_TEMPLATE.replace(
            "{{table_data_snippet}}", snippet
        )
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                top_p=0.95,
                presence_penalty=1,
                extra_body = {
                    "chat_template_kwargs": {'enable_thinking': True},
                    "top_k": 20,
                    "mip_p": 0,
                },
            )
            
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("LLM returned empty content")
            
            return {
                "input_item": (data_group, column_group),
                "pydantic_schema": self._extract_json(content),
                "error": None,
                "success": True,
            }
        except Exception as e:
            return {
                "input_item": (data_group, column_group),
                "pydantic_schema": None,
                "error": str(e),
                "success": False,
            }
        finally:
            semaphore.release()

    
    def _merge_schemas(self, group_schemas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple schemas into a single schema.
        
        Args:
            group_schemas: List of schemas from different groups
            
        Returns:
            Merged schema
        """
        merged_schema: Dict[str, Any] = {
            "title": "RowData",
            "type": "object",
        }
        
        # Merge properties from all schemas
        all_properties: Dict[str, Any] = {}
        for schema in group_schemas:
            if "properties" in schema:
                all_properties.update(schema["properties"])
        merged_schema["properties"] = all_properties
                
        # Merge required fields if they exist
        all_required = []
        for schema in group_schemas:
            if "required" in schema:
                all_required.extend(schema["required"])
        
        if all_required:
            merged_schema["required"] = list(set(all_required))
        
        return merged_schema

        
    async def transform_data(
        self,
        data_rows: List[List[Any]],
        formatted_header: List[str],
        pydantic_schema: Dict[str, Any],
        max_retries: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Transform all data rows in the table to the schema defined in pydantic_schema.
        """
        # Find common columns
        common_columns = self._find_common_columns(
            formatted_header=formatted_header,
            data_rows=data_rows,
            pydantic_schema=pydantic_schema,
        )
        # Reduce pydantic schema to only include non-common columns
        reduced_pydantic_schema = {k: v for k, v in pydantic_schema.items() if k != "properties"}
        reduced_pydantic_schema["properties"] = {k: v for k, v in pydantic_schema["properties"].items() if k not in common_columns}
        if "required" in reduced_pydantic_schema:
            reduced_pydantic_schema["required"] = [
                col for col in reduced_pydantic_schema["required"] 
                if col not in common_columns
            ]

        if reduced_pydantic_schema["properties"]:
            queue: List[Tuple[int, List[Any]]] = []
            for i, row in enumerate(data_rows):
                queue.append((i, row))
                
            results = []
            for attempt in range(max_retries):
                if not queue:
                    break
                    
                print(f"Attempt {attempt + 1}/{max_retries}: Processing {len(queue)} rows...")
                
                # Process batch
                batch_results = await self._transform_batch(
                    queue, 
                    formatted_header, 
                    reduced_pydantic_schema,
                )
                
                # Collect successes and prepare retries
                successful_results = []
                failed_items = []
                
                for res in batch_results:
                    if res["success"]:
                        successful_results.append(res)
                    else:
                        failed_items.append(res["input_item"])
                        
                results.extend(successful_results)
                queue = failed_items

            # Sort results by original index to maintain order
            results.sort(key=lambda x: x["input_item"][0])
        else:
            # If no properties, return empty rows
            results = [
                {
                    "success": True,
                    "error": None,
                    "transformed_row": {},
                    "input_item": (i, row)
                }
                for i, row in enumerate(data_rows)
            ]

        # Merge common columns
        for result in results:
            for col in common_columns:
                val_mapping = {k: v for k, v in zip(formatted_header, result["input_item"][1])}
                result["transformed_row"][col] = val_mapping[col]
        
        # Return just the transformed data
        return [result["transformed_row"] for result in results]


    async def _transform_batch(
        self,
        queue: List[Tuple[int, List[Any]]],
        formatted_header: List[str],
        pydantic_schema: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        tasks = [
            self._transform_single_row(index, row, formatted_header, pydantic_schema, semaphore)
            for index, row in queue
        ]
        return await tqdm_asyncio.gather(*tasks, desc="Transforming data")


    async def _transform_single_row(
        self,
        index: int,
        data_row: List[Any],
        formatted_header: List[str],
        pydantic_schema: Dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        await semaphore.acquire()
        try:
            raw_row = self._format_table_data_snippet_with_header(formatted_header, [data_row])
            prompt = TRANSFORM_DATA_TEMPLATE.replace(
                "{{raw_row}}", raw_row
            ).replace(
                "{{pydantic_schema}}", json.dumps(pydantic_schema, ensure_ascii=False)
            )
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                top_p=0.8,
                presence_penalty=1,
                extra_body = {
                    "chat_template_kwargs": {'enable_thinking': False},
                    "top_k": 20,
                    "mip_p": 0,
                },
                timeout=10,
            )
            
            content = response.choices[0].message.content
            if not content:
                 raise ValueError("Empty response from LLM")

            transformed_row = self._extract_json(content)
            if transformed_row.keys() != pydantic_schema["properties"].keys():
                raise ValueError("Transformed row keys do not match pydantic schema keys")
                 
            return {
                "success": True,
                "error": None,
                "transformed_row": transformed_row, # Parse JSON
                "input_item": (index, data_row),
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e), # Error message
                "transformed_row": None,
                "input_item": (index, data_row),
            }
        finally:
            semaphore.release()


    def _format_table_data_snippet(
        self,
        table_rows: List[List[Any]],
        sample_size: int = 3,
    ) -> str:
        """
        Format a snippet of the table data for the LLM prompt.
        Shows the first `sample_size` rows and the last `sample_size` rows.
        """
        table_data_snippet = "["
        total_rows = len(table_rows)
        
        # If table is small enough, just show the whole thing
        if total_rows <= (sample_size * 3):
            for i, row in enumerate(table_rows):
                table_data_snippet += str(row)
                if i < total_rows - 1:
                    table_data_snippet += ",\n "
            table_data_snippet += "]"
            return table_data_snippet

        # Format top rows
        top_rows = table_rows[:sample_size]
        table_data_snippet += str(top_rows[0])
        for row in top_rows[1:]:
            table_data_snippet += (",\n "+str(row))

        # Format middle rows
        middle_rows = random.sample(
            table_rows[sample_size:-sample_size], 
            sample_size
        )
        for row in middle_rows:
            table_data_snippet += (",\n "+str(row))
                
        # Format bottom rows
        bottom_rows = table_rows[-sample_size:]
        table_data_snippet += ",\n ...\n " + str(bottom_rows[0])
        for row in bottom_rows[1:]:
            table_data_snippet += (",\n "+str(row))
        table_data_snippet += "]"
        return table_data_snippet


    def _format_table_data_snippet_with_header(
        self,
        formatted_header: List[str],
        data_rows: List[List[Any]],
        sample_size: int = 5,
    ) -> str:
        """
        Format a snippet of the table data for the LLM prompt.
        Shows the header rows and the first `sample_size` rows.
        """
        if len(data_rows) == 1:
            return json.dumps(
                {column: val for column, val in zip(formatted_header, data_rows[0])}, ensure_ascii=False
            )
        else:
            sample_size = min(sample_size, len(data_rows))
            sample_rows = random.sample(data_rows, sample_size)
            return "\n".join([
                json.dumps(
                    {column: val for column, val in zip(formatted_header, sample_row)}, ensure_ascii=False
                )
                for sample_row in sample_rows
            ])


    def _extract_json(self, content: str) -> Dict[str, Any]:
        """
        Extract and parse the JSON block from the LLM response.
        """
        matches = self.json_pattern.findall(content)
        if not matches:
            # Fallback: try to find json block without code fences if explicitly requested
            # or just fail as per prompt requirements
            raise ValueError("No JSON block found in LLM response")
            
        # Take the last match as in the notebook
        json_str = matches[-1]
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from response: {e}")


    def _format_rows(self, rows: List[List[Any]]) -> str:
        """
        Format the rows for the LLM prompt.
        """
        if len(rows) == 0:
            return "[]"
        if len(rows) == 1:
            return str(rows[0])
        rows_snippet = "[" + str(rows[0])
        for row in rows[1:]:
            rows_snippet += ",\n " + str(row)
        rows_snippet += "]"
        return rows_snippet


    def _format_header_rows(self, header_rows: List[List[Any]]) -> List[str]:
        """
        Format the header rows for the LLM prompt.
        """
        return ["_".join(list(header_col)) for header_col in zip(*header_rows)]


    def _find_original_column_types(
        self, 
        formatted_header: List[str], 
        data_rows: List[List[Any]],
    ) -> Dict[str, str]:
        """
        Find the original column types from the data rows.
        """
        if not data_rows or not formatted_header:
            return {}
        
        column_types: Dict[str, str] = {}
        for col_name, col_values in zip(formatted_header, zip(*data_rows)):
            if all(val is None or isinstance(val, str) for val in col_values):
                column_types[col_name] = "string"
                continue
            elif all(val is None or isinstance(val, int) for val in col_values):
                column_types[col_name] = "integer"
                continue
            elif all(val is None or isinstance(val, float) for val in col_values):
                column_types[col_name] = "number"
                continue
            elif all(val is None or isinstance(val, bool) for val in col_values):
                column_types[col_name] = "boolean"
                continue
            else:
                column_types[col_name] = "unknown"
        
        return column_types


    def _find_common_columns(
        self, 
        formatted_header: List[str], 
        data_rows: List[List[Any]], 
        pydantic_schema: Dict[str, Any],
    ) -> List[str]:
        """
        Check if column names and types match between formatted header and pydantic schema.
        
        Args:
            formatted_header: List of original column names from the header
            data_rows: List of data rows to find original column types from
            pydantic_schema: New pydantic schema dictionary
            
        Returns:
            List of common columns if they exist, otherwise empty list
        """
        common_columns = []

        # First check if column names match
        schema_properties = pydantic_schema.get("properties", {})
        for col_name in formatted_header:
            if col_name in schema_properties.keys():
                common_columns.append(col_name)
        
        # If names match, check types
        original_column_types = self._find_original_column_types(formatted_header, data_rows)
        
        for col_name in common_columns:
            schema_type_str = schema_properties[col_name].get("type", "unknown")
            if schema_type_str == original_column_types[col_name]:
                continue
            else:
                common_columns.remove(col_name)
        
        return common_columns




# In design_schema method, instead of generating pydantic schema, generate SQL schema for different dialects like sqlite, mysql, postgresql, etc.
# In transform_data method, instead of generating a json object, generate a SQL query that can be used to insert the data into the database, then run the query to check if the data is inserted correctly.