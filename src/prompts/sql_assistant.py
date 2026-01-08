import json


MESSAGE_REWRITING_TEMPLATE = """
### Role
You are an expert Context Extractor for a database chatbot. Your task is to analyze a conversation between a "Customer" and a "Support Team" and identify the **background context** required to understand the Customer's LATEST message.

### Rules
- The context must be fully interpretable in isolation, requiring no access to the conversation history to understand. You must identify the core subject, ALL active references and constraints from the dialogue and synthesize them into the context. **Explicitly resolve** all pronouns and relative references by substituting them with the specific entities, dates, IDs, feature, values, etc. mentioned previously. The final context is **NOT ALLOWED** to contain any pronouns or vague references. 
- Only include the context that is related to the Customer's LATEST message. If there is no relevant context, return an empty string.
- The context must sound like the customer are re-describing the context to the support team.
- Output a JSON object inside a json markdown code block using this format:
```json
{{
    "context": "the relevant and specific context in Vietnamese"
}}
```

### Conversation:
{formatted_conversation}
""".strip()


TABLE_SELECTING_TEMPLATE = """
### Role:
You are a Database Schema Selector. Your task is to identify the subset of tables relevant to the user's LATEST request.

### Instructions:
1. Analyze the conversation between a "Customer" and a "Support Team" and identify "What is the background context required to understand the Customer's LATEST message?"
2. Review the **Table Summaries** and select tables that align with the background context and user's LATEST intent.
3. If the user asks a question that requires connecting data from multiple tables (e.g., "Customer names and their Order dates"), you MUST select ALL tables required to join them (e.g., `customers` AND `orders`).
4. Select only the tables that are strictly necessary. Do not include tables "just in case."
5. Return a valid JSON object inside a json code block using this format:
```json
{{
    "explanation": "Explanation of the decision in Vietnamese.",
    "relevant_tables": ["table_name_1", "table_name_2",...]
}}
```

### Table Summaries:
{table_summaries}

### Conversation:
{formatted_conversation}
""".strip()


SCHEMA_LINKING_TEMPLATE = """
You are an expert in SQL schema linking. 
Given a {dialect} table schema (DDL) and a conversation history, determine if the table is relevant to the latest customer query.

Your task:
1. Analyze the table schema and the conversation history. Focus on the latest customer message, using previous messages for context (e.g., to resolve references). Evaluate the Table Summary and Table Schema to see if the general topic matches the query. Answer "Y" (Yes) or "N" (No) regarding the table's relevance to the latest query.
2. If the answer is "Y", list ALL columns that are semantically related. 
   - You do NOT need to identify the exact columns for the final SQL query. 
   - You MUST include all columns that provide context, identifiers, or potential join keys related to the entities in the query.

Output must be a valid JSON object inside a json code block using this format:
```json
{{
    "is_related": "Y or N",
    "columns": ["column name 1", "column name 2"]
}}
```

Table Summary:
{table_summary}

Table Schema (DDL):
{table_info}

Conversation History:
{formatted_conversation}
""".strip()


SQL_GEN_TEMPLATE = """
### DATE INFORMATION:
Today is {date}

### Instructions:
You write SQL queries for a {dialect} database. The Support Team is querying the database to answer Customer questions, and your task is to assist by generating valid SQL queries strictly adhering to the database schema provided. Translate the latest customer message into a **single valid {dialect} query**, using the conversation history for context (e.g., resolving pronouns or follow-up filters).

### Guidelines:
1. Use only tables, columns, and relationships explicitly listed in the provided schema. Do not make assumptions about missing or inferred columns/tables.
2. Use only {dialect} syntax. Be aware that {dialect} has limited built-in date/time functions compared to other sql dialects. Do NOT use INSERT, UPDATE, DELETE, ALTER, DROP, CREATE, etc. statements. Only use SELECT statements to query the database.
3. Escape reserved keywords or case-sensitive identifiers using double quotes (" "), e.g., "order".
4. If the customer's question is ambiguous or unclear, you must make your best reasonable guess based on the schema. Ensure the query is optimized, precise, and error-free. 

### Output Format:
- Briefly explain your logic in Vietnamese. Identify the user's intent, how it relates to previous messages, and which tables/columns you selected. The write a valid SQL query in a sql markdown code block.
- Output format:
Reasoning: Reasoning of the decision in Vietnamese.
```sql
Your SQL query here.
```

**CRITICAL:** The code inside the ```sql ``` block is the **final answer** that will be executed. It must be precise and retrieve only the exact data needed—no more, no less.

**Note:** Unless the user is asking for a specific calculation (like "Count the total..."), you must always retrieve **all columns** (`SELECT *`) for the resulting records.

### Table Schema:
{table_infos}

### Conversation:
{formatted_conversation}
""".strip()


ANSWER_GEN_TEMPLATE = """
### THÔNG TIN NGÀY THÁNG:
Hôm nay là {date}

### NHIỆM VỤ:
Bạn là một trợ lý phân tích dữ liệu chuyên nghiệp. Nhiệm vụ của bạn là đưa ra câu trả lời bằng **Tiếng Việt** rõ ràng, chính xác và súc tích cho câu hỏi của người dùng, dựa hoàn toàn vào kết quả cơ sở dữ liệu (database results) được cung cấp.

**Lược đồ bảng (Table Schema)**:
{table_infos}

### CÁC NGUYÊN TẮC HƯỚNG DẪN:

1.  **Chính xác và Tuân thủ dữ liệu**:
    *   Câu trả lời phải dựa **TUYỆT ĐỐI** vào phần "Kết quả từ Database".
    *   Không được tự suy diễn hoặc đưa vào các kiến thức bên ngoài không có trong dữ liệu.
    *   Nếu kết quả trả về là rỗng (empty), hãy thông báo lịch sự rằng không tìm thấy dữ liệu phù hợp với yêu cầu.

2.  **Định dạng câu trả lời**:
    *   **Trả lời trực tiếp**: Đi thẳng vào vấn đề.
    *   **Danh sách/Bảng**: Nếu kết quả có nhiều dòng, hãy trình bày dưới dạng danh sách gạch đầu dòng hoặc bảng Markdown cho dễ đọc.
    *   **Số liệu tổng hợp**: Nếu kết quả là một con số duy nhất (tổng, đếm, trung bình), hãy viết thành một câu hoàn chỉnh (Ví dụ: "Tổng doanh thu là 50.000.000 VNĐ").

3.  **Trình bày dữ liệu (Formatting)**:
    *   **Con số**: Sử dụng dấu phân cách hàng nghìn (ví dụ: 1.000 hoặc 1,000 tùy theo ngữ cảnh, nhưng phải nhất quán).
    *   **Tiền tệ**: Thêm đơn vị tiền tệ phù hợp nếu có (ví dụ: VNĐ, $, USD).
    *   **Ngày tháng**: Chuyển đổi sang định dạng ngày tháng Tiếng Việt tự nhiên (ví dụ: "Ngày 01 tháng 01 năm 2024").

4.  **Ngữ cảnh và Thuật ngữ**:
    *   Sử dụng "Truy vấn SQL" để hiểu ngữ cảnh lọc dữ liệu (ví dụ: nếu SQL có `WHERE status = 'active'`, hãy nói rõ đây là các đơn hàng có trạng thái là "đang hoạt động").
    *   Sử dụng ngôn ngữ kinh doanh/đời thường. **Không** nhắc đến tên bảng kỹ thuật (như `tbl_users`, `col_price`) hoặc cú pháp code trong câu trả lời cuối cùng.

5.  **Văn phong**:
    *   Chuyên nghiệp, khách quan và hữu ích.
    *   Tránh các câu máy móc như mà hãy trả lời tự nhiên như một con người.

**Đầu ra**:
Chỉ xuất ra câu trả lời cuối cùng bằng Tiếng Việt (sử dụng Markdown).
""".strip()


QUERY_DATABASE_TOOL = json.dumps({
    'type': 'function',
    'function': {
        'name': 'query_database',
        'description': 'Thực hiện câu truy vấn {{dialect}} và trả về kết quả',
        'parameters': {
            'properties': {
                'sql_query': {
                    'description': 'Câu truy vấn {{dialect}}',
                    'type': 'string'
                }
            },
            'required': ['sql_query'],
            'type': 'object'
        }
    }
})


RERANK_ROWS_TEMPLATE = """
### Role
You are an expert Database Row Filter and Re-ranker. Your task is to analyze a conversation and a list of candidate rows retrieved from the table `{table_name}`. You must identify which rows strictly satisfy the user's intent and constraints.

### Table Info
{table_info}

### Rules
1. Analyze Intent: Read the **Conversation** to understand what the "Customer" is looking for. Pay attention to the **LATEST** message but use previous messages to resolve context (filters, exact values, numerical ranges, categories).
2. Verify Rows: The "Candidate Rows" section contains a batch of raw data and many of these are **NOISE** You must verify the data in each row against the user's constraints.
3. Strict ID Extraction: You must only return the `rowid` of rows that are relevant. If a row is ambiguous or does not match, ignore it. 
4. Output Format: Output a single JSON object inside a json markdown code block.
```json
{{
    "reasoning": "Reasoning of the decision",
    "relevant_row_ids": [ row_id_1, row_id_2, ... ] # list of row ids as integers
}}
```
If no rows match, return an empty list `[]`.

### Conversation
{formatted_conversation}

### Candidate Rows
{formatted_rows}
""".strip()