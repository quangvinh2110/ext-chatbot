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
    *   Process string columns by splitting or reformatting them into multiple, more structured columns. E.g., If a `Location` column contains "New York, NY", you should split it into a `city` (string) and a `state` (string) column. Similarly, if a price column contains "3000 usd", split it into `price_value` (float) and `price_currency` (string).
    *   Add a concise description to each field in the schema explaining its purpose or the transformation applied.

### Output Format:

You must return **strictly** a valid JSON object within a ```json ... ``` block, with no markdown formatting or conversational text. The format must be a standard Pydantic `model_json_schema()` output:

```json
{
  "title": "RowData",
  "type": "object",
  "properties": {
      "column_1_name": {
          "type": "string",
          "description": "Description of the column 1."
      },
      "column_2_name": {
           "type": "number",
           "description": "Description of the column 2 after transformation."
      },
      ...
  },
  "required": ["column_1_name", "column_2_name"]
}
```

### HEADER:
{{header}}

### DATA SNIPPET:
{{table_data_snippet}}
""".strip()


EXTRACT_STRUCTURE_TEMPLATE = """
You are an expert Data Engineer and Python Pydantic Specialist. Your task is to analyze a raw snippet of tabular data (comprising the first few rows and the last few rows of a dataset) and generate a data cleaning specification.

**Input Data:**
I will provide you with a list of lists representing the top and bottom rows of a parsed Excel table.
Example Input format:
```
[[item_1_1, item_1_2],
 [item_2_1, item_2_2], 
 ... 
 [item_n_1, item_n_2]]
```

**Your Goal:**
Output a JSON object containing two main parts: `table_structure` (identifying headers/footers) and `pydantic_schema` (a Pydantic-compliant JSON schema for the data).

### Instructions:

1.  Analyze Table Structure (Header/Footer):
    *   Output: Identify which rows are headers and which rows are footers. Return the indices of the header rows and the indices of the footer rows. If header or footer is missing, return `null` for `header_indices` or `null` for `footer_indices`.

2.  Design a Robust Schema (Data Cleaning and Transformation):
    *   Look at the values in the rows to determine the best data type for a database or downstream analysis. Avoid using `string` for everything unless there is no other way. For example, if the value is a date, use `datetime` instead of `string`, if the value is a number, use `float` or `int` instead of `string`.
    *   Crucially, you must actively define a schema that prepares the data for subsequent analysis phases. You should:
        *   Process string columns by splitting or reformatting (or both) those columns. E.g., If a value in a price column is saved as "3000 usd" (string), you should split it into a "price_value" (float) and a "price_currency" (string) column.
        *   Add a concise description to each column in the schema explaining its purpose or the transformation applied.
        *   If the raw data snippet clearly lacks a header (i.e., the first row is data), generate a new, descriptive header, and define the schema based on these new column names.

### Output Format:

You must return **strictly** a valid JSON object within a ```json ... ``` block, with no markdown formatting or conversational text. The format must be:

```json
{
  "table_structure": {
    "header_indices": [integer, integer, ...], // list of indices of header rows, or null
    "footer_indices": [integer, integer, ...], // list of indices of footer rows, or null
  },
  "pydantic_schema": {
    // Insert standard Pydantic model_json_schema() output here
    "title": "RowData",
    "type": "object",
    "properties": { 
        // Define your new columns here, using descriptive keys and proper types.
        "column_1_name": {
            "type": "string",
            "description": "Description of the column 1"
        },
        ...
    },
  }
}
```

### Input Data:
{{table_data_snippet}}
""".strip()


TRANSFORM_DATA_TEMPLATE = """
You are an expert Data Engineer specializing in data cleaning and transformation. Your task is to process a single row of raw data, apply transformations defined by a provided Pydantic schema, and output the result as a clean JSON object that strictly adheres to the schema's structure and data types.

**Input Data:**
You will receive three inputs:
1.  **`RAW_HEADER`**: The list of strings identified as the header row(s) in the original table. Use this to interpret the meaning of columns in the `RAW_ROW`. If no header was identified, this will be `null`.
2.  **`RAW_ROW`**: A list of values representing a single row from the original table that needs transformation.
3.  **`PYDANTIC_SCHEMA`**: The JSON schema defining the desired structure, data types, and transformation logic for the output row.

**Your Goal:**
Output a JSON object containing the transformed row.

### Instructions:
For each property defined in the `PYDANTIC_SCHEMA["properties"]`:
    *   Locate the corresponding value(s) in the `RAW_ROW`, referencing the `RAW_HEADER` if needed for context.
    *   Apply the necessary cleaning, parsing, or splitting logic implicitly described by the new column name, expected data type, and the column description in the schema.
    *   Ensure the final value adheres precisely to the specified Pydantic type.

### Output Format:
Return **strictly** a single JSON object, with no markdown formatting or conversational text. The format must be:
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