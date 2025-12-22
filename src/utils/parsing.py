import re
import json

_sql_markdown_re = re.compile(r"```sql\s*([\s\S]*?)\s*```", re.DOTALL)
_json_markdown_re = re.compile(r"```json\s*([\s\S]*?)\s*```", re.DOTALL)

def parse_sql_output(msg_content: str) -> str:
    try:
        match = _sql_markdown_re.search(msg_content)
        if match:
            return match.group(1).strip()
        else:
            raise ValueError("No SQL query found in the content")
    except Exception:
        raise ValueError("Failed to parse SQL from response")


def parse_json_output(msg_content: str) -> dict:
    try:
        match = _json_markdown_re.search(msg_content)
        if match:
            return json.loads(match.group(1).strip())
        else:
            raise ValueError("No JSON found in the content")
    except Exception:
        raise ValueError("Failed to parse JSON from response")