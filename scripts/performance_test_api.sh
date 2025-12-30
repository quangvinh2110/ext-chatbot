#!/bin/bash

# With custom input file and concurrency
python performance_test.py \
  --input /path/to/input.jsonl \
  --base-url http://localhost:8000 \
  --endpoint /sql_search \
  --max-concurrent 50