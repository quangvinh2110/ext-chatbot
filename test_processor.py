import numpy as np
import json
from src.parser.excel_parser import TableInfo
from src.processor.table_processor import TableProcessor
from src.utils import load_env

def test_processor():
    # Load env
    load_env(".env")
    
    # Create mock table data
    # Header: Region, Branch, Address
    # Data: HCM, Branch 1, Address 1
    data = np.array([
        ["Region", "Branch", "Address"],
        ["HCM", "Branch 1", "Address 1"],
        ["HCM", "Branch 2", "Address 2"],
        ["HN", "Branch 3", "Address 3"],
    ], dtype=object)
    
    table = TableInfo(
        data=data,
        start_row=0,
        end_row=3,
        start_col=0,
        num_cols=3,
        num_data_rows=3,
        merged_cells=[]
    )
    
    print("Initializing TableProcessor...")
    try:
        processor = TableProcessor()
    except ValueError as e:
        print(f"Skipping test because configuration is missing: {e}")
        return

    print("Processing table...")
    try:
        result = processor.process_table(table)
        print("Result:")
        print(json.dumps(result, indent=2))
        
        # Basic validation
        assert "table_structure" in result
        assert "pydantic_schema" in result
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        # Print full exception
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_processor()

