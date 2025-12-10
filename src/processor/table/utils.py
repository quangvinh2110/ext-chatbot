import re
from typing import List, Any, Dict
import random
import json
from difflib import SequenceMatcher


def format_table_data_snippet(
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


def format_table_data_snippet_with_header(
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


def format_header_rows(
    header_rows: List[List[Any]],
) -> List[str]:
    """
    Format the header rows for the LLM prompt.
    """
    return ["_".join(list(header_col)) for header_col in zip(*header_rows)]


JSON_PATTERN = re.compile(r"```json\n(.*?)\n```", re.DOTALL)
def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract and parse the JSON block from the LLM response.
    """
    matches = JSON_PATTERN.findall(text)
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


def compute_lcs_length(s1: str, s2: str) -> int:
    """
    Finds the length of the longest common substring between two strings using difflib.
    """
    seq_match = SequenceMatcher(None, s1, s2)
    match = seq_match.find_longest_match(0, len(s1), 0, len(s2))

    if match.size != 0:
        return len(s1[match.a : match.a + match.size])
    else:
        return 0
