import os
from typing import Optional, List, Any
from openai import OpenAI
from ...prompts import GENERATE_HEADER_TEMPLATE
from .utils import format_table_data_snippet_with_header, extract_json


class HeaderGenerator:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.client = OpenAI(
            base_url=base_url or os.getenv("LLM_BASE_URL"),
            api_key=api_key or os.getenv("LLM_API_KEY")
        )
        if not (model or os.getenv("LLM_MODEL")):
            raise ValueError("LLM_MODEL not set. Please provide a model name or set LLM_MODEL environment variable.")
        self.model: str = str(model) if model else str(os.getenv("LLM_MODEL"))


    def __call__(
        self,
        table_rows: List[List[Any]],
        max_retries: int = 3,
    ) -> List[str]:
        snippet = format_table_data_snippet_with_header(
            formatted_header=[f"col{i}" for i in range(len(table_rows[0]))],
            data_rows=table_rows,
        )
        prompt = GENERATE_HEADER_TEMPLATE.replace(
            "{{table_data_snippet}}", snippet
        )
        print(table_rows)
        exit()
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
                generated_header = extract_json(content)
                if "generated_header" not in generated_header or not generated_header["generated_header"]:
                    raise ValueError("Generated header not found in response")
                elif len(generated_header["generated_header"]) != len(table_rows[0]):
                    raise ValueError("Generated header does not match the number of columns in the table")
                return generated_header["generated_header"]
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                continue
                
        raise ValueError("Failed to process request after max retries")
