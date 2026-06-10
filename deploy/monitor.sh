#!/bin/bash
# deploy/monitor.sh — Blackwell server monitoring
# Usage: ./deploy/monitor.sh [--continuous]

set -e

SERVER_URL="${SERVER_URL:-http://localhost:8123}"
INTERVAL="${INTERVAL:-5}"

echo "=== Blackwell Server Monitor ==="
echo "Server: $SERVER_URL"
echo "Interval: ${INTERVAL}s"
echo ""

check_health() {
    local response=$(curl -s "$SERVER_URL/health" 2>/dev/null)
    if [ -z "$response" ]; then
        echo "[$(date '+%H:%M:%S')] ERROR: Server not responding"
        return 1
    fi
    
    local status=$(echo "$response" | jq -r '.status' 2>/dev/null || echo "unknown")
    local model=$(echo "$response" | jq -r '.model' 2>/dev/null || echo "unknown")
    local gpu_used=$(echo "$response" | jq -r '.gpu_used_mb' 2>/dev/null || echo "0")
    local gpu_total=$(echo "$response" | jq -r '.gpu_total_mb' 2>/dev/null || echo "0")
    local requests=$(echo "$response" | jq -r '.requests' 2>/dev/null || echo "0")
    local errors=$(echo "$response" | jq -r '.errors' 2>/dev/null || echo "0")
    local latency=$(echo "$response" | jq -r '.avg_latency_ms' 2>/dev/null || echo "0")
    local uptime=$(echo "$response" | jq -r '.uptime_sec' 2>/dev/null || echo "0")
    
    local gpu_pct=$((gpu_used * 100 / gpu_total))
    local error_rate="0"
    if [ "$requests" -gt 0 ]; then
        error_rate=$(echo "scale=2; $errors * 100 / $requests" | bc 2>/dev/null || echo "0")
    fi
    
    printf "[%s] STATUS=%-4s GPU=%d/%dMB (%d%%) REQUESTS=%d ERRORS=%d (%.1f%%) LATENCY=%.1fms UPTIME=%ds\n" \
        "$(date '+%H:%M:%S')" \
        "$status" \
        "$gpu_used" \
        "$gpu_total" \
        "$gpu_pct" \
        "$requests" \
        "$errors" \
        "$error_rate" \
        "$latency" \
        "$uptime"
    
    return 0
}

# Test mode (single check)
if [ "$1" != "--continuous" ]; then
    check_health
    exit $?
fi

# Continuous mode
echo "Press Ctrl+C to stop"
echo ""
while true; do
    check_health || true
    sleep "$INTERVAL"
done