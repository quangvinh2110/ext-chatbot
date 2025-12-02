import asyncio
import numpy as np
import json
import os
from src.parser.excel_parser import TableInfo
from src.processor.table_processor import TableProcessor
from src.utils import load_env

async def test_transform():
    load_env(".env")
    
    if not os.getenv("LLM_API_KEY"):
        print("Skipping test: LLM_API_KEY not set")
        return

    # Mock data
    data = np.array([
        ["Product", "Price", "Currency"],
        ["Apple", "1.0", "USD"],
        ["Banana", "0.5", "USD"]
    ], dtype=object)
    
    table = TableInfo(
        data=data,
        start_row=0, end_row=2, start_col=0, num_cols=3, num_data_rows=2, merged_cells=[]
    )
    
    # Mock structure info from Step 1
    structure_info = {
        "table_structure": {
            "header_indices": [0],
            "footer_indices": []
        },
        "pydantic_schema": {
            "title": "ProductPrice",
            "type": "object",
            "properties": {
                "product_name": {"type": "string", "description": "Name of the product"},
                "price_value": {"type": "number", "description": "Price value"},
                "currency_code": {"type": "string", "description": "Currency code"}
            },
            "required": ["product_name", "price_value", "currency_code"]
        }
    }
    
    try:
        processor = TableProcessor()
        
        print("Testing transform_data...")
        results = await processor.transform_data(table, structure_info)
        print(json.dumps(results, indent=2))
        
        assert len(results) == 2
        assert results[0]["product_name"] == "Apple"
        # Allow float comparison
        assert abs(results[0]["price_value"] - 1.0) < 0.001
        print("Test passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_transform())

