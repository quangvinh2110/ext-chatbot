IDENTIFY_HEADER_FOOTER_TEMPLATE = """
You are an expert Data Engineer specializing in table parsing. Your task is to analyze the provided raw snippet of tabular data, which consists of the top and bottom rows of a larger dataset. Based on the content, identify which rows belong to the logical header (containing column names or meta-information) and which rows belong to the logical footer (containing summaries or total counts). Return the 0-based indices of these rows. If no clear header or footer is present, return `null` for the respective index list.

**Input Data:**
I will provide you with a list of lists representing the top and bottom rows of a parsed Excel table.
Example Input format:
```
[[item_1_1, item_1_2],
 [item_2_1, item_2_2], 
 ... 
 [item_n_1, item_n_2]]
```

**Output Format:**
Return **strictly** a single JSON object, with no markdown formatting or conversational text. The format must be:
```json
{
  "header_indices": [integer, integer, ...], // list of indices of header rows, or null
  "footer_indices": [integer, integer, ...], // list of indices of footer rows, or null
}
```

### Data Snippet:
{{table_data_snippet}}
""".strip()


DESIGN_SCHEMA_TEMPLATE = """
You are an expert Data Engineer and Python Pydantic Specialist. Your task is to analyze a raw snippet of tabular data and its associated header, then design a new schema for the next data analysis phase.

**Input Data:**
I will provide you two inputs:
1.   HEADER: A list of lists of strings representing the table header (can be multi-level), or `null` if no header is present.
2.   DATA SNIPPET: A list of lists representing the top and bottom rows of the table data.

**Your Goal:**
Output a single Pydantic-compliant JSON schema that prepares the data for the next data analysis phase.

### Instructions:
*   Look at the values in the `data snippet` rows to determine the best data type for a database or downstream analysis. Avoid using `string` for everything unless there is no other way. For example, if a value is a date, use `datetime` with a specific format; if it's a number, use `float` or `int`.
*   Crucially, you must actively define a schema that prepares the data for subsequent analysis. You should:
    *   Process string columns by splitting or reformatting them into multiple, more structured columns. E.g., If a `Địa điểm` column contains "phường Tân Thuận, Quận 7, Hồ Chí Minh", you should split it into a `Địa điểm_Thành phố` (string), a `Địa điểm_Phường` (string) and a `Địa điểm_Quận` (string) column. Similarly, if a price column contains "3000 usd", split it into `price_value` (float) and `price_currency` (string). Note that you can add new columns to the schema, but you MUST keep the original column as well.
    *   Add a concise Vietnamese description to each field in the schema explaining its purpose or the transformation applied. The description must contain the data type and 3-5 examples of the values in the column.

### Output Format:

You must return **strictly** a valid JSON object within a ```json ... ``` block, with no markdown formatting or conversational text. The format must be a standard Pydantic `model_json_schema()` output:

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

### HEADER:
{{header}}

### DATA SNIPPET:
{{table_data_snippet}}
""".strip()


TRANSFORM_DATA_TEMPLATE = """
You are an expert Data Engineer specializing in data cleaning and transformation. Your task is to process a single row of raw data, apply transformations defined by a provided Pydantic schema, and output the result as a clean JSON object that strictly adheres to the schema's structure and data types.

**Input Data:**
You will receive four inputs:
1.  **`RAW_HEADER`** (Optional): The list of strings identified as the header row(s) in the original table. Use this to interpret the meaning of columns in the `RAW_ROW`. If no header was identified, this will be `null`.
2.  **`RAW_ROW`**: A list of values representing a single row from the original table that needs transformation.
3.  **`PYDANTIC_SCHEMA`**: The JSON schema defining the desired structure, data types, and transformation logic for the output row.

**Your Goal:**
Output a JSON object containing the transformed row within a ```json ... ``` block.

### Instructions:
For each property defined in the `PYDANTIC_SCHEMA["properties"]`:
    *   Locate the corresponding value(s) in the `RAW_ROW`, referencing the `RAW_HEADER` if needed for context.
    *   Apply the necessary cleaning, parsing, or splitting logic implicitly described by the new column name, expected data type, and the column description in the schema.
    *   Ensure the final value adheres precisely to the specified Pydantic type.

### Output Format:
Return your final JSON object within a ```json ... ``` block. The format must be:
```json
{
  "new_column_1_name": transformed_value_1,
  "new_column_2_name": transformed_value_2,
  // ... all properties from the schema must be present
}
```

### Input Data:

**RAW_HEADER:**
{{raw_header}}

**RAW_ROW:**
{{raw_row}}

**PYDANTIC_SCHEMA:**
{{pydantic_schema}}
""".strip()