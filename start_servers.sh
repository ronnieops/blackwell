#!/bin/bash
# start_servers.sh — Start multiple Blackwell inference servers
# Usage: ./start_servers.sh [model...]
# Without args: starts all models

cd /mnt/data/dev/projects/blackwell

# Kill existing servers
pkill -f "http_subprocess" 2>/dev/null || true
sleep 1

# Model configurations: name port weight_dir
MODELS=(
    "llama32-1b:8123:/mnt/data/ai/models/llama32-1b-int4"
    "llama31-8b:8124:/mnt/data/ai/models/llama31-8b-int4"
)

start_model() {
    local name=$1
    local port=$2
    local wdir=$3
    
    echo "Starting $name on port $port with weights: $wdir"
    cd "$wdir"
    /mnt/data/dev/projects/blackwell/server/http_subprocess "$port" "$name" &
    cd - > /dev/null
    sleep 2
}

if [ $# -eq 0 ]; then
    # Start all models
    for m in "${MODELS[@]}"; do
        IFS=':' read -r name port wdir <<< "$m"
        start_model "$name" "$port" "$wdir"
    done
else
    # Start specified models
    for arg in "$@"; do
        for m in "${MODELS[@]}"; do
            IFS=':' read -r name port wdir <<< "$m"
            if [[ "$name" == "$arg" || "$arg" == "all" ]]; then
                start_model "$name" "$port" "$wdir"
                break
            fi
        done
    done
fi

echo "Servers started. Check with: curl http://localhost:8123/health"