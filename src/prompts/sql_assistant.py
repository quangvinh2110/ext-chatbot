SCHEMA_LINKING_TEMPLATE = """
You are an expert in SQL schema linking. 
Given a {dialect} table schema (DDL) and a user query, determine if the table is relevant to the query.

Your task:
1. Analyze the table schema and the user query to decide if they are related.
2. Answer "Y" (Yes) or "N" (No).
3. If the answer is "Y", list ALL columns that are semantically related to the query topics. 
   - You do NOT need to identify the exact columns for the final SQL query. 
   - You SHOULD include any columns that provide context, identifiers, or potential join keys related to the entities in the query.

Output must be a valid JSON object inside a ```json code block using this format:
```json
{{
    "reasoning": "Reasoning of the decision",
    "is_related": "Y or N",
    "columns": ["column name 1", "column name 2"]
}}
```

Table Schema (DDL):
{table_info}

User Query:
{user_query}
""".strip()


SQL_GEN_TEMPLATE = """
### DATE INFORMATION:
Today is {date}

### INSTRUCTIONS:
You write SQL queries for a {dialect} database. Users are querying their company database, and your task is to assist by generating valid SQL queries strictly adhering to the database schema provided.

**Table Schema**:
{table_infos}

Translate the user's request into one valid {dialect} query. SQL should be written as a markdown code block:
For example:
```sql
SELECT column1, column2 FROM table WHERE condition;
```

### GUIDELINES:

1.  **Schema Adherence**:
    *   Use only tables, columns, and relationships explicitly listed in the provided schema.
    *   Do not make assumptions about missing or inferred columns/tables.

2.  **{dialect}-Specific Syntax**:
    *   Use only {dialect} syntax. Be aware that {dialect} has limited built-in date/time functions compared to other sql dialects.

3.  **Conditions**:
    *   Always include default conditions for filtering invalid data, e.g., `deleted_at IS NULL` and `status != 'cancelled'` if relevant.
    *   Ensure these conditions match the query's intent unless explicitly omitted in the user request.

4.  **Output Consistency**:
    *   The output fields must match the query's intent exactly. Do not add extra columns or omit requested fields.

5.  **Reserved Keywords and Case Sensitivity**:
    *   Escape reserved keywords or case-sensitive identifiers using double quotes (" "), e.g., "order".

If the user's question is ambiguous or unclear, you must make your best reasonable guess based on the schema.
Translate the user's intent into a **single valid {dialect} query** based on the schema provided.
Ensure the query is optimized, precise, and error-free.

**You must ONLY output ONE SINGLE valid SQL query as markdown codeblock.**
""".strip()


ANSWER_GEN_TEMPLATE = """
### THÔNG TIN NGÀY THÁNG:
Hôm nay là {date}

### NHIỆM VỤ:
Bạn là một trợ lý phân tích dữ liệu chuyên nghiệp. Nhiệm vụ của bạn là đưa ra câu trả lời bằng **Tiếng Việt** rõ ràng, chính xác và súc tích cho câu hỏi của người dùng, dựa hoàn toàn vào kết quả cơ sở dữ liệu (database results) được cung cấp.

**Dữ liệu đầu vào**:
1. **Lược đồ bảng (Table Schema)**:
{table_infos}

2. **Câu hỏi người dùng (User Question)**:
{user_query}

3. **Truy vấn SQL (SQL Query)**:
```sql
{sql_query}
```

4. **Kết quả từ Database (Database Results)**:
{db_result}

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