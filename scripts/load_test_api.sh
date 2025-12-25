#!/bin/bash

# Run load test with tag filtering
# python load_test.py resources/logs/exports_1766646264633-lf-traces-export-cmio1vjfs000ynv0795mpsunw.jsonl --filter-tag guru

# Or with other options
python load_test.py resources/logs/exports_1766646264633-lf-traces-export-cmio1vjfs000ynv0795mpsunw.jsonl \
    --filter-tag guru \
    --endpoint /sql_search \
    --concurrency 20 \
    --base-url http://localhost:8321