import re

_sql_markdown_re = re.compile(r"```sql\s*([\s\S]*?)\s*```", re.DOTALL)


def parse_sql_output(msg_content: str) -> str:
    try:
        match = _sql_markdown_re.search(msg_content)
        if match:
            return match.group(1).strip()
        else:
            raise ValueError("No SQL query found in the content")
    except Exception:
        return msg_content