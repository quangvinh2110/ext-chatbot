import argparse
import os
import openpyxl

from src.parser.excel_parser import ExcelParser
from src.processor.table import TableProcessor
from src.utils import write_json

from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "env", "internal.env"))


def process_excel(input_file: str, output_dir: str):
    """
    Process an Excel file: parse tables and transform data.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Processing file: {input_file}")
    
    # Get sheet names
    try:
        wb = openpyxl.load_workbook(input_file, read_only=True)
        sheet_names = wb.sheetnames
        wb.close()
    except Exception as e:
        print(f"Error reading workbook: {e}")
        return

    processor = TableProcessor()

    with ExcelParser(input_file, skip_empty_rows=False) as parser:
        for sheet_name in sheet_names[1:6]:
            print(f"Processing sheet: {sheet_name}")
            try:
                tables = parser.parse_sheet(sheet_name)
            except Exception as e:
                print(f"Error parsing sheet {sheet_name}: {e}")
                continue

            if not tables:
                print(f"No tables found in sheet: {sheet_name}")
                continue

            for i, table in enumerate(tables):
                print(f"Processing table {i+1}/{len(tables)} in sheet {sheet_name}...")
                
                try:
                    result = processor(
                        table,
                        max_retries=100
                    )
                    transformed_data = result.get("transformed_data", [])
                    
                    if not transformed_data:
                        print(f"Warning: No data transformed for table {i+1} in {sheet_name}")
                        continue

                    # Determine output filename
                    if len(tables) == 1:
                        filename = f"{sheet_name}.json"
                    else:
                        filename = f"{sheet_name}_{i}.json"
                    
                    # Sanitize filename
                    filename = "".join([c for c in filename if c.isalpha() or c.isdigit() or c in (' ', '.', '_', '-')]).strip()
                    output_path = os.path.join(output_dir, filename)
                    
                    write_json(result, output_path)
                    print(f"Saved to {output_path}")

                except Exception as e:
                    print(f"Error processing table {i+1} in sheet {sheet_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an Excel file to extract and transform tables.")
    parser.add_argument("--input_file", type=str, help="Path to the Excel file")
    parser.add_argument("--output_dir", type=str, help="Directory to save output JSON files")
    
    args = parser.parse_args()
    
    process_excel(args.input_file, args.output_dir)

