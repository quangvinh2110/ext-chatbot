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


def extract_fn(text: str) -> tuple[str, str]:
    """Extract function name and arguments from tool call text."""
    fn_name, fn_args = '', ''
    fn_name_s = '"name": "'
    fn_name_e = '", "'
    fn_args_s = '"arguments": '
    
    i = text.find(fn_name_s)
    k = text.find(fn_args_s)
    
    if i > 0:
        _text = text[i + len(fn_name_s):]
        j = _text.find(fn_name_e)
        if j > -1:
            fn_name = _text[:j]
    
    if k > 0:
        fn_args = text[k + len(fn_args_s):]
    
    fn_args = fn_args.strip()
    if len(fn_args) > 2:
        fn_args = fn_args[:-1]
    else:
        fn_args = ''
    
    return fn_name, fn_args