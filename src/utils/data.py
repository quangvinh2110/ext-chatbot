from difflib import SequenceMatcher

def pydantic_to_sqlite_type(pydantic_type: str) -> str:
    type_mapping = {
        'string': 'TEXT',
        'integer': 'INTEGER',
        'number': 'REAL',
        'boolean': 'INTEGER',  # SQLite uses INTEGER for booleans (0 or 1)
        'null': 'TEXT',
    }
    return type_mapping.get(pydantic_type.lower(), 'TEXT')


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