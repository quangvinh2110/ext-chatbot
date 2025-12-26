import asyncio
import json
import time
import argparse
import statistics
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

import httpx
from tqdm import tqdm


@dataclass
class RequestResult:
    """Result of a single request"""
    request_id: int
    endpoint: str
    success: bool
    status_code: Optional[int] = None
    response_time: float = 0.0
    error_message: Optional[str] = None
    response_data: Optional[Dict[str, Any]] = None


@dataclass
class LoadTestStats:
    """Statistics from load test"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: List[float] = field(default_factory=list)
    status_codes: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    errors: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def avg_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    
    @property
    def median_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return statistics.median(self.response_times)
    
    @property
    def p95_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        index = int(len(sorted_times) * 0.95)
        return sorted_times[index] if index < len(sorted_times) else sorted_times[-1]
    
    @property
    def p99_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        index = int(len(sorted_times) * 0.99)
        return sorted_times[index] if index < len(sorted_times) else sorted_times[-1]
    
    @property
    def min_response_time(self) -> float:
        return min(self.response_times) if self.response_times else 0.0
    
    @property
    def max_response_time(self) -> float:
        return max(self.response_times) if self.response_times else 0.0


def load_jsonl(file_path: Path, filter_tag: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load JSONL file and return list of JSON objects.
    
    Supports two formats:
    1. Direct format: {"current_message": "...", "conversation_history": [...]}
    2. Langfuse export format: {"input": "...", "metadata": {"history": "..."}, "tags": [...]}
    
    Args:
        file_path: Path to JSONL file
        filter_tag: If provided, only include lines where this tag is in the tags array
        
    Returns:
        List of JSON objects in API request format
    """
    requests = []
    skipped_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                
                # Check if this is Langfuse export format (has "input" field)
                if "input" in data:
                    # Filter by tag if specified
                    if filter_tag:
                        tags = data.get("tags", [])
                        if filter_tag not in tags:
                            skipped_count += 1
                            continue
                    
                    # Extract input as current_message
                    current_message = data.get("input", "")
                    if not current_message:
                        skipped_count += 1
                        continue
                    
                    # Parse conversation history from metadata.history
                    conversation_history = None
                    metadata = data.get("metadata", {})
                    if metadata and "history" in metadata:
                        history_value = metadata["history"]
                        if isinstance(history_value, str):
                            try:
                                conversation_history = json.loads(history_value)
                            except json.JSONDecodeError:
                                # If parsing fails, leave as None
                                conversation_history = None
                        elif isinstance(history_value, list):
                            conversation_history = history_value
                    
                    # Create request payload
                    request_data = {
                        "current_message": current_message,
                        "conversation_history": conversation_history
                    }
                    requests.append(request_data)
                else:
                    # Direct format - use as-is
                    if filter_tag:
                        # Check tags if present
                        tags = data.get("tags", [])
                        if filter_tag not in tags:
                            skipped_count += 1
                            continue
                    requests.append(data)
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                continue
    
    if skipped_count > 0:
        print(f"Filtered out {skipped_count} lines (tag filter: {filter_tag})")
    
    return requests


async def make_request(
    client: httpx.AsyncClient,
    base_url: str,
    endpoint: str,
    request_data: Dict[str, Any],
    request_id: int,
    timeout: float = 300.0
) -> RequestResult:
    """
    Make a single HTTP request to the API.
    
    Args:
        client: HTTP client
        base_url: Base URL of the API
        endpoint: API endpoint (e.g., '/sql_search' or '/route')
        request_data: Request payload
        request_id: Unique identifier for this request
        timeout: Request timeout in seconds
        
    Returns:
        RequestResult object
    """
    url = f"{base_url.rstrip('/')}{endpoint}"
    start_time = time.time()
    
    try:
        response = await client.post(
            url,
            json=request_data,
            timeout=timeout
        )
        response_time = time.time() - start_time
        
        try:
            response_data = response.json()
        except Exception:
            response_data = {"text": response.text}
        
        success = 200 <= response.status_code < 300
        
        return RequestResult(
            request_id=request_id,
            endpoint=endpoint,
            success=success,
            status_code=response.status_code,
            response_time=response_time,
            error_message=None if success else f"HTTP {response.status_code}",
            response_data=response_data
        )
    except httpx.TimeoutException:
        response_time = time.time() - start_time
        return RequestResult(
            request_id=request_id,
            endpoint=endpoint,
            success=False,
            response_time=response_time,
            error_message="Request timeout"
        )
    except Exception as e:
        response_time = time.time() - start_time
        return RequestResult(
            request_id=request_id,
            endpoint=endpoint,
            success=False,
            response_time=response_time,
            error_message=str(e)
        )


async def run_load_test(
    base_url: str,
    endpoint: str,
    requests: List[Dict[str, Any]],
    concurrency: int = 10,
    timeout: float = 300.0
) -> LoadTestStats:
    """
    Run load test with concurrent requests.
    
    Args:
        base_url: Base URL of the API
        endpoint: API endpoint
        requests: List of request payloads
        concurrency: Number of concurrent requests
        timeout: Request timeout in seconds
        
    Returns:
        LoadTestStats object
    """
    stats = LoadTestStats()
    stats.total_requests = len(requests)
    
    semaphore = asyncio.Semaphore(concurrency)
    
    async def bounded_request(request_data: Dict[str, Any], request_id: int) -> RequestResult:
        await semaphore.acquire()
        try:
            async with httpx.AsyncClient() as client:
                return await make_request(
                    client, base_url, endpoint, request_data, request_id, timeout
                )
        except Exception:
            # Re-raise the exception after releasing semaphore
            raise
        finally:
            semaphore.release()
    
    # Create tasks for all requests
    tasks = [
        bounded_request(request_data, idx)
        for idx, request_data in enumerate(requests)
    ]
    
    # Execute all requests with progress bar
    results = []
    with tqdm(total=len(tasks), desc="Running load test") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            pbar.update(1)
    
    # Collect statistics
    for result in results:
        if result.success:
            stats.successful_requests += 1
        else:
            stats.failed_requests += 1
            if result.error_message:
                stats.errors[result.error_message] += 1
        
        stats.response_times.append(result.response_time)
        
        if result.status_code:
            stats.status_codes[result.status_code] += 1
    
    return stats


def print_stats(stats: LoadTestStats, endpoint: str):
    """Print load test statistics."""
    print("\n" + "="*60)
    print(f"Load Test Results for {endpoint}")
    print("="*60)
    print(f"Total Requests:     {stats.total_requests}")
    print(f"Successful:         {stats.successful_requests} ({stats.success_rate:.2f}%)")
    print(f"Failed:             {stats.failed_requests} ({100 - stats.success_rate:.2f}%)")
    print("\nResponse Times (seconds):")
    print(f"  Average:          {stats.avg_response_time:.3f}s")
    print(f"  Median:           {stats.median_response_time:.3f}s")
    print(f"  Min:              {stats.min_response_time:.3f}s")
    print(f"  Max:              {stats.max_response_time:.3f}s")
    print(f"  P95:              {stats.p95_response_time:.3f}s")
    print(f"  P99:              {stats.p99_response_time:.3f}s")
    
    if stats.status_codes:
        print("\nStatus Codes:")
        for status_code, count in sorted(stats.status_codes.items()):
            percentage = (count / stats.total_requests) * 100
            print(f"  {status_code}: {count} ({percentage:.2f}%)")
    
    if stats.errors:
        print("\nErrors:")
        for error, count in sorted(stats.errors.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error}: {count}")
    
    print("="*60 + "\n")


async def check_health(base_url: str) -> bool:
    """Check if the API is healthy."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url.rstrip('/')}/health", timeout=10.0)
            return response.status_code == 200
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Load test the SQL Search API using requests from a JSONL file"
    )
    parser.add_argument(
        "jsonl_file",
        type=Path,
        help="Path to JSONL file containing test requests (one JSON object per line)"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        choices=["/sql_search", "/route"],
        default="/sql_search",
        help="API endpoint to test (default: /sql_search)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent requests (default: 10)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Request timeout in seconds (default: 300)"
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip health check before running tests"
    )
    parser.add_argument(
        "--filter-tag",
        type=str,
        default=None,
        help="Filter lines by tag (e.g., 'guru'). Only lines with this tag in the tags array will be included."
    )
    
    args = parser.parse_args()
    
    # Validate JSONL file exists
    if not args.jsonl_file.exists():
        print(f"Error: JSONL file not found: {args.jsonl_file}")
        return 1
    
    # Check API health
    if not args.skip_health_check:
        print(f"Checking API health at {args.base_url}...")
        health_ok = asyncio.run(check_health(args.base_url))
        if not health_ok:
            print("Warning: API health check failed. Continuing anyway...")
        else:
            print("API is healthy!")
    
    # Load requests from JSONL file
    print(f"Loading requests from {args.jsonl_file}...")
    if args.filter_tag:
        print(f"Filtering by tag: {args.filter_tag}")
    requests = load_jsonl(args.jsonl_file, filter_tag=args.filter_tag)
    
    if not requests:
        print("Error: No valid requests found in JSONL file")
        return 1
    
    print(f"Loaded {len(requests)} requests")
    print(f"Endpoint: {args.endpoint}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Timeout: {args.timeout}s")
    print("\nStarting load test...")
    
    # Run load test
    start_time = time.time()
    stats = asyncio.run(
        run_load_test(
            base_url=args.base_url,
            endpoint=args.endpoint,
            requests=requests,
            concurrency=args.concurrency,
            timeout=args.timeout
        )
    )
    total_time = time.time() - start_time
    
    # Print statistics
    print_stats(stats, args.endpoint)
    print(f"Total test time: {total_time:.2f}s")
    print(f"Requests per second: {stats.total_requests / total_time:.2f}")
    
    return 0 if stats.success_rate > 0 else 1


if __name__ == "__main__":
    exit(main())

