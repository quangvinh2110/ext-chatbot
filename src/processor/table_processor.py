import re
import json
import os
import random
import asyncio
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from openai import OpenAI, AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from ..parser.excel_parser import TableInfo
from ..prompts import EXTRACT_STRUCTURE_TEMPLATE, TRANSFORM_DATA_TEMPLATE
from ..utils import set_seed


set_seed(42)


class TableProcessor:
    """
    Processes tables extracted by ExcelParser using an LLM to determine structure and schema.
    """
    
    def __init__(self, 
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 max_concurrent_requests: int = 10):
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

    async def __call__(
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
        print("Extracting table structure...")
        structure_info = self.extract_table_structure(table, max_retries)
        
        print("Transforming data...")
        # transformed_data = asyncio.run(self.transform_data(
        #     table, 
        #     structure_info, 
        #     max_retries
        # ))
        transformed_data = await self.transform_data(
            table, 
            structure_info, 
            max_retries
        )
        
        return {
            "structure_info": structure_info,
            "transformed_data": transformed_data
        }

    def extract_table_structure(
        self,
        table: TableInfo,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Process a single table to extract structure and schema information.
        
        Args:
            table: TableInfo object from ExcelParser.
            max_retries (int, optional): Maximum number of retries for API call and JSON parsing. Defaults to 3.
            
        Returns:
            Dict containing 'table_structure' and 'pydantic_schema'.
            
        Raises:
            ValueError: If LLM response is empty or JSON cannot be parsed after retries.
        """
        snippet = self._format_table_data_snippet(table.data)
        
        prompt = EXTRACT_STRUCTURE_TEMPLATE.replace(
            "{{table_data_snippet}}",
            snippet
        )
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
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
                    
                return self._extract_json(content)
            
            except Exception as e:
                last_exception = e
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                continue
                
        # If we get here, all retries failed
        if last_exception:
            raise last_exception
        else:
            raise ValueError("Failed to process table after max retries")

    async def transform_data(
        self,
        table: TableInfo,
        structure_info: Dict[str, Any],
        max_retries: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Transform all data rows in the table to the schema defined in structure_info.
        """
        data_list = table.data.tolist()
        table_structure = structure_info["table_structure"]
        header_indices = table_structure.get("header_indices") or []
        footer_indices = table_structure.get("footer_indices") or []
        
        # Get header rows for context
        header_rows = [data_list[i] for i in header_indices]
        formatted_header = self._format_rows(header_rows)
        
        pydantic_schema = structure_info["pydantic_schema"]
        
        # Identify rows to process
        rows_to_process = []
        for i, row in enumerate(data_list):
            if i not in header_indices and i not in footer_indices:
                rows_to_process.append((i, row))
                
        if not rows_to_process:
            return []
            
        # Process with retries
        results = []
        queue = rows_to_process
        
        for attempt in range(max_retries):
            if not queue:
                break
                
            print(f"Attempt {attempt + 1}/{max_retries}: Processing {len(queue)} rows...")
            
            # Process batch
            batch_results = await self._transform_batch(
                queue, 
                formatted_header, 
                pydantic_schema
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
            
            if not queue:
                break
                
        # Sort results by original index to maintain order
        results.sort(key=lambda x: x["input_item"][0])
        
        # Return just the transformed data
        return [r["transformed_data"] for r in results]

    async def _transform_batch(
        self,
        batch_items: List[Tuple[int, List[Any]]],
        formatted_header: str,
        pydantic_schema: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        tasks = [
            self._transform_single_row(index, row, formatted_header, pydantic_schema, semaphore)
            for index, row in batch_items
        ]
        return await tqdm_asyncio.gather(*tasks, desc="Transforming data")

    async def _transform_single_row(
        self,
        index: int,
        row: List[Any],
        formatted_header: str,
        pydantic_schema: Dict[str, Any],
        semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        await semaphore.acquire()
        try:
            row_snippet = self._format_rows([row])
            
            prompt = TRANSFORM_DATA_TEMPLATE.replace(
                "{{raw_header}}", formatted_header
            ).replace(
                "{{raw_row}}", row_snippet
            ).replace(
                "{{pydantic_schema}}", str(pydantic_schema)
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
                    "guided_json": pydantic_schema
                },
            )
            
            content = response.choices[0].message.content
            if not content:
                 raise ValueError("Empty response from LLM")
                 
            return {
                "success": True,
                "error": None,
                "transformed_data": json.loads(content), # Parse JSON
                "input_item": (index, row),
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e), # Error message
                "transformed_data": None,
                "input_item": (index, row),
            }
        finally:
            semaphore.release()

    def _format_table_data_snippet(
        self,
        table_data: np.ndarray,
        sample_size: int = 3,
    ) -> str:
        """
        Format a snippet of the table data for the LLM prompt.
        Shows the first `sample_size` rows and the last `sample_size` rows.
        """
        table_data_snippet = "["
        rows = table_data.tolist()
        total_rows = len(rows)
        
        # If table is small enough, just show the whole thing
        if total_rows <= (sample_size * 3):
            for i, row in enumerate(rows):
                table_data_snippet += str(row)
                if i < total_rows - 1:
                    table_data_snippet += ",\n "
            table_data_snippet += "]"
            return table_data_snippet

        # Format top rows
        top_rows = rows[:sample_size]
        table_data_snippet += str(top_rows[0])
        for row in top_rows[1:]:
            table_data_snippet += (",\n "+str(row))

        # Format middle rows
        middle_rows = random.sample(
            rows[sample_size:-sample_size], 
            sample_size
        )
        for row in middle_rows:
            table_data_snippet += (",\n "+str(row))
                
        # Format bottom rows
        bottom_rows = rows[-sample_size:]
        table_data_snippet += ",\n ...\n " + str(bottom_rows[0])
        for row in bottom_rows[1:]:
            table_data_snippet += (",\n "+str(row))
        table_data_snippet += "]"
        return table_data_snippet

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
