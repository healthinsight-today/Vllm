#!/bin/bash

# Memory and performance monitoring script for vLLM deployment

PORT=${1:-8000}
INTERVAL=${2:-5}

echo "🔍 Monitoring vLLM memory usage and performance"
echo "Port: $PORT | Update interval: ${INTERVAL}s"
echo "Press Ctrl+C to stop"
echo ""

# Create log file with timestamp
LOG_FILE="vllm_monitoring_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOG_FILE"
echo ""

# Header
printf "%-19s | %-8s | %-12s | %-12s | %-8s | %-8s | %-10s\n" \
    "Timestamp" "GPU_Mem" "GPU_Util" "CPU_Usage" "Requests" "Status" "Response"
echo "$(printf '%.0s-' {1..80})"

while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # GPU metrics
    gpu_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "N/A")
    gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "N/A")
    
    # CPU usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}' | cut -d'%' -f1 2>/dev/null || echo "N/A")
    
    # Service status
    if systemctl is-active --quiet vllm-qwen.service 2>/dev/null; then
        service_status="Running"
    else
        service_status="Stopped"
    fi
    
    # Test response time
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        start_time=$(date +%s.%N)
        response=$(curl -s -X POST "http://localhost:$PORT/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d '{"model": "qwen2.5-32b", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 5}' \
            --max-time 10 2>/dev/null)
        end_time=$(date +%s.%N)
        
        if echo "$response" | grep -q "choices" 2>/dev/null; then
            response_time=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "0")
            response_time=$(printf "%.2fs" "$response_time")
        else
            response_time="Error"
        fi
        
        # Count active connections (approximate)
        active_requests=$(ss -tulpn | grep ":$PORT" | wc -l 2>/dev/null || echo "0")
    else
        response_time="N/A"
        active_requests="0"
    fi
    
    # Display current metrics
    printf "%-19s | %-8s | %-12s | %-12s | %-8s | %-8s | %-10s\n" \
        "$timestamp" "${gpu_memory}MB" "${gpu_util}%" "${cpu_usage}%" \
        "$active_requests" "$service_status" "$response_time"
    
    # Log to file
    echo "$timestamp,$gpu_memory,$gpu_util,$cpu_usage,$active_requests,$service_status,$response_time" >> "$LOG_FILE"
    
    sleep $INTERVAL
done