import os
from typing import Optional, Dict, Any, List
from openai import OpenAI
from ...prompts import IDENTIFY_HEADER_FOOTER_TEMPLATE
from .utils import format_table_data_snippet, extract_json


class HeaderFooterIdentifier:
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
        sheet_name: Optional[str] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        snippet = format_table_data_snippet(table_rows)
        prompt = IDENTIFY_HEADER_FOOTER_TEMPLATE.replace(
            "{{table_data_snippet}}",
            snippet
        ).replace(
            "{{sheet_name}}", sheet_name or "Unknown"
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
                
                header_footer_info = extract_json(content)
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
