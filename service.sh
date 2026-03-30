#!/bin/bash
set -e

NUM_WORKERS=1
BASE_PORT=8000
echo $CNB_VSCODE_PROXY_URI
for i in $(seq 1 $NUM_WORKERS); do
    PORT=$((BASE_PORT+i)) # 计算每个服务的端口号
    PORT=$PORT python ./service/app.py &
    sleep 10 # 给每个服务一些时间来启动
done

echo "ALL $NUM_WORKERS started"