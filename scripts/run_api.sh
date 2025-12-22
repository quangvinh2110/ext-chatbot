#!/bin/bash

# set_proxy

nohup uvicorn api:app --host 0.0.0.0 --port 8321 > server.log 2>&1 &

echo "Server started. Logs are in server.log"