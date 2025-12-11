IDENTIFY_HEADER_FOOTER_TEMPLATE = """
You are an expert Data Engineer. Your task is to analyze a data snippet to identify header/footer rows, and if no header exists, to generate a logical one.

### Definitions:

1.  What is a Header?
    *   A row is considered belong to the header if it contains only **column titles/labels**. If a row contains both column titles/labels and column values, then it is not a header.
    *   **The "JSON Key" Test:** After deciding which rows belong to the header  (or not), you must check if the column titles/labels can be used as keys for the data rows that follow (you can form a JSON object from the column titles/labels and the data rows that follow and check if the JSON object is valid). If yes, then the row belongs to the header. For multi-level headers, the titles are combined by underscores to form a single key.

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
  "reasoning": "Reasoning of the decision",
  "header_indices": [list of indices of header rows],  // or null
  "footer_indices": [list of indices of footer rows], // or null
}
```

### Sheet Name:
{{sheet_name}}

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

### Sheet Name:
{{sheet_name}}

### Table Data Snippet:
{{table_data_snippet}}
""".strip()


GROUP_COLUMNS_TEMPLATE = """
You are an expert Data Architect. Your task is to group columns from the provided data snippet into semantically related clusters.

### Core Principle:
Group columns that must be analyzed together. The most critical rule is to place **all** columns with **overlapping information** in the same group. This allows for coherent data extraction, and avoid duplication after data extraction phase.

### Instructions:
- Analyze the column names and their corresponding values in the `Data Snippet`.
- Carefully look at values of each column to identify all columns that have overlapping information and form groups.
- Assign every column to exactly one group.

### Output Format:
Return a single JSON object inside a ```json ... ``` block.
```json
{
  "explanation": "Explanation of your grouping logic",
  "column_groups": [
    {
      "group_name": "name of the group",
      "columns": ["column1", "column2", "column3"]
    },
    ...
  ]
}
```

### Sheet Name:
{{sheet_name}}

### Data Snippet:
{{table_data_snippet}}
""".strip()


MERGE_COLUMN_GROUPS_TEMPLATE = """
You are an expert Data Architect. Your task is to merge column groups from the provided data snippet into semantically related bigger groups.

### Core Principle:
- Merge groups to form new bigger groups that contain columns that must be analyzed together. 
- You are only allowed to merge column groups. If two columns are in the same group before merging, then you have to keep them in the same new group after merging.

### Instructions:
- Analyze the column names and their corresponding values in the `Data Snippet`.
- Carefully look at values of each column to identify all columns that must be analyzed together and form new groups.
- Each new group should contain **at least {{group_min_size}} columns** and **at most {{group_max_size}} columns**.
- Assign every column to exactly one new group after merging.

### Output Format:
Return a single JSON object inside a ```json ... ``` block.
```json
{
  "reasoning": "Reasoning of the decision",
  "column_groups": [
    {
      "group_name": "name of the group",
      "columns": ["column1", "column2",...]
    },
    ...
  ]
}

### Sheet Name:
{{sheet_name}}

### Original Column Groups:
{{column_groups}}

### Data Snippet:
{{table_data_snippet}}
""".strip()


DESIGN_SCHEMA_TEMPLATE = """
You are an expert Data Engineer and Python Pydantic Specialist. Your task is to analyze a raw snippet of tabular data, then design a new, enhanced schema for a data analysis phase.

**Input and Output:**
You will receive a `Data snippet`, which contains a list of JSON objects, each represent a single row from the table, with column titles as keys. Output a Pydantic-compliant JSON schema that prepares the data for the next data analysis phase.

### Instructions:
*   Scope of Analysis: Your decisions must be based exclusively on the data within the Data snippet. Treat this snippet as the complete source of truth. Do not speculate about what the rest of the dataset might look like.
*   Look at each column in the `Data snippet` to determine the best data type for a database or downstream analysis based on the column's semantic meaning. For instance, if a column represent calendar dates or timestamps (e.g., "dd/mm/yyyy"), use a `datetime` type; if they represent quantitative data, use `float` or `int`. Avoid defaulting to `string` if a more specific type is suitable.
*   The Component Extraction Rule: Look for string columns that contain complex, multi-part data, and determine if you can reliably extract common, meaningful components from the majority of the rows for the next analysis phase. For each found component, you must verify that component is consistent across ALL rows and that component is useful. 
*   Analysis-Driven Restructuring: Before creating new columns from an existing one by extracting a component, you must justify it with analytical utility. Ask yourself, "Does the current format of the original column prevent or complicate common analytical operations like filtering, sorting, or mathematical aggregation?". Then ask yourself "Adding this new column can make the data more useful for the next analysis phase?".
*   If you create a new column by extracting a component from an existing one, you must name it following the pattern: `original column name` + underscore + `component name`.
*   You still need to keep all the original columns (maybe with a different data type) in the schema.
*   For every column (both original and new), add a concise Vietnamese description to explain the column's purpose. Besides, include approximately 3 example values.

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
          "description": "Mô tả của cột 1.",
          "examples": ["Ví dụ 1", "Ví dụ 2", "Ví dụ 3"],
      },
      "Tên cột 2": {
           "type": "number",
           "description": "Mô tả của cột 2",
           "examples": ["Ví dụ 1", "Ví dụ 2", "Ví dụ 3"],
      },
      ...
  },
}
```

### Sheet Name:
{{sheet_name}}

### Data Snippet:
{{table_data_snippet}}
""".strip()


TRANSFORM_DATA_TEMPLATE = """
You are a Data Engineer. Your task is to transform a single `Raw Row` of data into a clean JSON object that conforms to the provided `Pydantic Schema`.

### Core Task:
Map the values from `Raw Row` to the properties defined in the `Pydantic Schema`. `Raw Row` is a dictionary. Use the dictionary keys (column titles in the original header) to intelligently match values to the new schema's fields.

After mapping, ensure every value in the final JSON strictly matches the data type and example values specified in the schema.

### Output Format:
Return only a single JSON object inside a ```json ... ``` code block.

### Input Data:

**Raw Row:**
{{raw_row}}

**Pydantic Schema:**
{{pydantic_schema}}
""".strip()