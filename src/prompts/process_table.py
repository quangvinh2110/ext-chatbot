IDENTIFY_HEADER_FOOTER_TEMPLATE = """
You are an expert Data Engineer. Your task is to analyze a data snippet to identify header/footer rows, and if no header exists, to generate a logical one.

### Definitions:

1.  What is a Header?
    *   A row is considered belong to the header if it contains only **column titles/labels** (e.g., `['First Name', 'Last Name', 'Email']`). If a row contains both label and value, then it is not a header.
    *   **The "JSON Key" Test:** After deciding which rows belong to the header, you must check if the column titles/labels can be used as keys for the data rows that follow. If yes, then the row belongs to the header. For multi-level headers, the titles are combined by underscores to form a single key.

2.  What is a Footer?
    *   A footer contains summary data like totals, averages, or page numbers.

### Instructions:
- Analyze the `Data Snippet` using the definitions above.
- Return the 0-based indices for any header and footer rows.
- If no clear header or footer exists, return `null`.

### Output Format:
Return a single JSON object inside a ```json ... ``` block.
```json
{
  "explanation": "Explanation of the decision",
  "header_indices": [list of indices of header rows],  // or null
  "footer_indices": [list of indices of footer rows], // or null
}
```

### Data Snippet:
{{table_data_snippet}}
""".strip()


GENERATE_HEADER_TEMPLATE = """
You are an expert Data Analyst. Your task is to generate a descriptive header for a dataset that is missing one.

You will be given a snippet of data as a list of dictionaries. The keys in these dictionaries are generic placeholders. Your goal is to replace them by inferring the true meaning of each column from its values.

### Instructions:
1.  Analyze Each Column: For each generic key (e.g., `col0`, `col1`, ...), examine the values associated with it across all the provided rows.
2.  Infer a Title: Based on your analysis, create a short, descriptive title for the column. The number of titles you generate must exactly match the number of keys provided.

### Output Format:
Return a single JSON object inside a ```json ... ``` block.
```json
{
  "explanation": "Explanation of the decision",
  "generated_header": ["title_for_col0", "title_for_col1", ...]
}
```

### Table Data Snippet:
{{table_data_snippet}}
""".strip()


DESIGN_SCHEMA_TEMPLATE = """
You are an expert Data Engineer and Python Pydantic Specialist. Your task is to analyze a raw snippet of tabular data, then design a new, enhanced schema for a data analysis phase.

**Input Data:**
You will receive a `DATA_SNIPPET`, which contains a list of JSON objects, each represent a single row from the table, with column titles as keys.

**Your Goal:**
Output a Pydantic-compliant JSON schema that prepares the data for the next data analysis phase.

### Instructions:
*   Scope of Analysis: Your decisions must be based exclusively on the data within the DATA_SNIPPET. Treat this snippet as the complete source of truth. Do not speculate about what the rest of the dataset might look like.
*   Look at each column in the `DATA_SNIPPET` to determine the best data type for a database or downstream analysis. For instance, if a column's values are dates (e.g., "dd/mm/yyyy"), use a `datetime` type; if they are numeric, use `float` or `int`. Avoid defaulting to `string` if a more specific type is suitable.
*   Restructure for Analysis:
    *   Identify Patterns: Look for string columns that contain complex, multi-part data.
    *   The Component Extraction Rule: Determine if you can reliably extract common, meaningful components from the majority of the rows. For each found component, you must verify that component is consistent across ALL rows. If only a few values have the same component while the majority do not, the component is not general enough. In this case, do NOT split the column. But if all values have the same component, then you can create a new column with the component as the value.
    *   Preserve Originals: When you do split a column, you must keep the original column in the schema in addition to the new, smaller columns you create.
*   For every field (both original and new), add a concise Vietnamese description. The description must explain the field's purpose and include approximately 3 example values from the data.

### Output Format:
Output an explanation of the decision and a valid JSON object within a ```json ... ``` block:

Explanation: # your explanation of the decision here
```json
{
  "title": "RowData",
  "type": "object",
  "properties": {
      "Tên cột 1": {
          "type": "string",
          "description": "Mô tả của cột 1."
      },
      "Tên cột 2": {
           "type": "number",
           "description": "Mô tả của cột 2"
      },
      ...
  },
  "required": ["Tên cột 1", "Tên cột 2"]
}
```

### DATA SNIPPET:
{{table_data_snippet}}
""".strip()


TRANSFORM_DATA_TEMPLATE = """
You are a Data Engineer. Your task is to transform a single `RAW_ROW` of data into a clean JSON object that conforms to the provided `PYDANTIC_SCHEMA`.

### Core Task:
Map the values from `RAW_ROW` to the properties defined in the `PYDANTIC_SCHEMA`. `RAW_ROW` is a dictionary. Use the dictionary keys (column titles in the original header) to intelligently match values to the new schema's fields.

After mapping, ensure every value in the final JSON strictly matches the data type and example values specified in the schema.

### Output Format:
Return only a single JSON object inside a ```json ... ``` code block.

### Input Data:

**RAW_ROW:**
{{raw_row}}

**PYDANTIC_SCHEMA:**
{{pydantic_schema}}
""".strip()