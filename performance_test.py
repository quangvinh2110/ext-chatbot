"""
Script to process test data from JSONL file and call SQL search API
"""
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import httpx
from tqdm import tqdm


async def call_api(
    client: httpx.AsyncClient,
    base_url: str,
    endpoint: str,
    current_message: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> tuple[Dict[str, Any], float]:
    """
    Call the API endpoint and measure response time.
    
    Args:
        client: HTTP async client
        base_url: Base URL of the API
        endpoint: API endpoint (e.g., '/sql_search' or '/route')
        current_message: Current user message
        conversation_history: Optional conversation history
        
    Returns:
        Tuple of (response_data, response_time_in_seconds)
    """
    url = f"{base_url.rstrip('/')}{endpoint}"
    
    # Validate current_message is not empty (required field)
    if not current_message or not current_message.strip():
        return {
            "error": "ValidationError",
            "detail": "current_message is required and cannot be empty"
        }, 0.0
    
    payload = {
        "current_message": current_message,
        "conversation_history": conversation_history
    }
    
    start_time = datetime.now()
    response_data: Dict[str, Any] = {}
    try:
        response = await client.post(url, json=payload, timeout=300000.0)
        
        # Check status code and handle errors
        if response.status_code >= 400:
            # Try to parse error response as JSON for better error details
            try:
                error_detail = response.json()
            except Exception:
                error_detail = response.text
            
            response_data = {
                "error": f"HTTP {response.status_code}",
                "detail": error_detail,
                "status_code": response.status_code
            }
        else:
            response_data = response.json()
    except httpx.TimeoutException:
        response_data = {
            "error": "TimeoutException",
            "detail": "Request timed out"
        }
    except Exception as e:
        response_data = {
            "error": str(type(e).__name__),
            "detail": str(e)
        }
    finally:
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
    
    return response_data, response_time


async def process_test_file(
    input_file: str,
    output_file: str,
    base_url: str = "http://localhost:8000",
    endpoint: str = "/sql_search",
    max_concurrent: int = 5
):
    """
    Process test data from JSONL file and call API for each entry.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        base_url: Base URL of the API
        endpoint: API endpoint (e.g., '/sql_search' or '/route')
        max_concurrent: Maximum number of concurrent requests
    """
    # Read all lines from input file
    print(f"Reading input file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()][:2]
    
    print(f"Found {len(lines)} entries to process")
    print(f"Endpoint: {endpoint}")
    print(f"Base URL: {base_url}")
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_entry(entry_data: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Process a single entry with rate limiting"""
        async with semaphore:
            current_message = entry_data.get("current_message", "")
            conversation_history = entry_data.get("conversation_history")
            
            # Validate and log if there's an issue
            if not current_message:
                print(f"Warning: Entry {index} has empty current_message")
            
            async with httpx.AsyncClient() as client:
                response_data, response_time = await call_api(
                    client=client,
                    base_url=base_url,
                    endpoint=endpoint,
                    current_message=current_message,
                    conversation_history=conversation_history
                )
            
            # Combine input and output data
            result = {
                "input": {
                    "current_message": current_message,
                    "conversation_history": conversation_history
                },
                "output": response_data,
                "response_time_seconds": response_time,
                "index": index
            }
            return result
    
    # Process all entries
    results = []
    tasks = []
    
    for idx, line in enumerate(lines):
        try:
            entry_data = json.loads(line)
            tasks.append(process_entry(entry_data, idx))
        except json.JSONDecodeError as e:
            print(f"Warning: Skipping line {idx + 1} due to JSON decode error: {e}")
            results.append({
                "input": {"raw_line": line},
                "output": {"error": "JSON decode error", "detail": str(e)},
                "response_time_seconds": 0.0,
                "index": idx
            })
    
    # Execute all tasks with progress bar
    print(f"Processing {len(tasks)} entries with max {max_concurrent} concurrent requests...")
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
        result = await coro
        results.append(result)
    
    # Sort results by index to maintain order
    def get_index(result: Dict[str, Any]) -> int:
        idx = result.get("index", 0)
        return int(idx) if isinstance(idx, (int, str)) else 0
    
    results.sort(key=get_index)
    
    # Write results to output file
    print(f"Writing results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # Print summary statistics
    def has_error(result: Dict[str, Any]) -> bool:
        output = result.get("output", {})
        return isinstance(output, dict) and "error" in output
    
    successful = sum(1 for r in results if not has_error(r))
    failed = len(results) - successful
    
    def get_response_time(result: Dict[str, Any]) -> float:
        rt = result.get("response_time_seconds", 0.0)
        if isinstance(rt, (int, float)):
            return float(rt)
        return 0.0
    
    response_times = [get_response_time(r) for r in results]
    avg_time = sum(response_times) / len(response_times) if response_times else 0.0
    total_time = sum(response_times)
    
    print("\n" + "="*50)
    print("Summary:")
    print(f"  Total entries: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Average response time: {avg_time:.2f} seconds")
    print(f"  Total processing time: {total_time:.2f} seconds")
    print("="*50)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process test data and call SQL search API")
    parser.add_argument(
        "--input",
        type=str,
        default="/Users/vinhnguyen/Projects/ext-chatbot/resources/logs/batdongsan_test.jsonl",
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/vinhnguyen/Projects/ext-chatbot/resources/logs/",
        help="Directory to save output file"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        choices=["/sql_search", "/route"],
        default="/sql_search",
        help="API endpoint to test (default: /sql_search)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum number of concurrent requests (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Generate output filename with datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"results_{timestamp}.jsonl")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run async processing
    asyncio.run(process_test_file(
        input_file=args.input,
        output_file=output_file,
        base_url=args.base_url,
        endpoint=args.endpoint,
        max_concurrent=args.max_concurrent
    ))
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

