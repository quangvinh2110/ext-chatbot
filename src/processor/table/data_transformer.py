import json
import os
from typing import Optional, Dict, Any, List, Tuple
from openai import AsyncOpenAI
import asyncio
from tqdm.asyncio import tqdm_asyncio

from ...prompts import TRANSFORM_DATA_TEMPLATE
from .utils import format_table_data_snippet_with_header, extract_json


class DataTransformer:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.client = AsyncOpenAI(
            base_url=base_url or os.getenv("LLM_BASE_URL"),
            api_key=api_key or os.getenv("LLM_API_KEY")
        )
        if not (model or os.getenv("LLM_MODEL")):
            raise ValueError("LLM_MODEL not set. Please provide a model name or set LLM_MODEL environment variable.")
        self.model: str = str(model) if model else str(os.getenv("LLM_MODEL"))


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



    async def __call__(
        self,
        data_rows: List[List[Any]],
        formatted_header: List[str],
        pydantic_schema: Dict[str, Any],
        max_retries: int = 3,
        max_concurrent_requests: int = 10,
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
        print(reduced_pydantic_schema)

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
                    max_concurrent_requests=max_concurrent_requests,
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
        max_concurrent_requests: int = 10,
    ) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(max_concurrent_requests)
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
            raw_row = format_table_data_snippet_with_header(formatted_header, [data_row])
            prompt = TRANSFORM_DATA_TEMPLATE.replace(
                "{{raw_row}}", raw_row
            ).replace(
                "{{pydantic_schema}}", json.dumps(pydantic_schema, ensure_ascii=False)
            )
            response = await self.client.chat.completions.create(
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

            transformed_row = extract_json(content)
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
