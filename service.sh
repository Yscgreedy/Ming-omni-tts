#!/bin/bash
NUM_WORKERS=8
BASE_PORT=8000
for i in $(seq 1 $NUM_WORKERS); do
    PORT=$((BASE_PORT+i))
    python ./service/app.py --port $PORT &
done

echo "ALL $NUM_WORKERS started"