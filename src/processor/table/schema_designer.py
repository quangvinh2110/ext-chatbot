import os
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from ...prompts import DESIGN_SCHEMA_TEMPLATE
from .utils import format_table_data_snippet_with_header, extract_json


class SchemaDesigner:
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

    async def __call__(
        self,
        data_groups: List[List[List[Any]]],
        column_groups: List[List[str]],
        max_retries: int = 3,
        max_concurrent_requests: int = 10,
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
            batch_results = await self._design_schema_for_groups(
                queue=queue,
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

    
    async def _design_schema_for_groups(
        self,
        queue: List[Tuple[List[List[Any]], List[str]]],
        max_concurrent_requests: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Design schema for each column group asynchronously.
        
        Args:
            queue: List of data groups and column groups
            
        Returns:
            List of schemas, one for each group
        """
        semaphore = asyncio.Semaphore(max_concurrent_requests)
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

        snippet = format_table_data_snippet_with_header(
            formatted_header=column_group,
            data_rows=data_group,
            sample_size=10,
        )
        prompt = DESIGN_SCHEMA_TEMPLATE.replace(
            "{{table_data_snippet}}", snippet
        )
        try:
            response = await self.client.chat.completions.create(
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

            pydantic_schema = extract_json(content)
            for column in column_group:
                if column not in pydantic_schema["properties"]:
                    raise ValueError(f"Column {column} not found in pydantic schema")
            new_columns = [col for col in pydantic_schema["properties"] if col not in column_group]
            for derived_column in new_columns:
                flag = False
                for original_column in column_group:
                    if original_column.replace(" ", "_") in derived_column.replace(" ", "_"):
                        flag = True
                        break
                if not flag:
                    raise ValueError(f"Derived column {derived_column} is not derived from any original column")
            
            return {
                "input_item": (data_group, column_group),
                "pydantic_schema": pydantic_schema,
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

    
    def _merge_schemas(
        self, 
        group_schemas: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
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
                
        return merged_schema
