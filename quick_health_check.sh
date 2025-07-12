#!/bin/bash

check_vllm_health() {
    local port=${1:-8000}
    local timeout=${2:-30}
    
    echo "Checking vLLM health on port $port..."
    
    for i in $(seq 1 $timeout); do
        if curl -s http://localhost:$port/health > /dev/null 2>&1; then
            echo "✅ vLLM is healthy on port $port"
            
            # Quick inference test
            response=$(curl -s -X POST "http://localhost:$port/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -d '{"model": "qwen2.5-32b", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 5}' \
                | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['choices'][0]['message']['content'] if 'choices' in data else 'error')" 2>/dev/null)
            
            if [ "$response" != "error" ] && [ "$response" != "" ]; then
                echo "✅ Inference test passed: $response"
                return 0
            else
                echo "⚠️ Health check passed but inference failed"
                return 1
            fi
        fi
        echo -n "."
        sleep 1
    done
    
    echo "❌ vLLM health check failed after $timeout seconds"
    return 1
}

# Main execution
case "${1:-single}" in
    "multi")
        echo "🔍 Checking multiple vLLM instances..."
        check_vllm_health 8000 30  # Production
        check_vllm_health 8001 30  # Staging  
        check_vllm_health 8002 30  # Development
        ;;
    "single")
        port=${2:-8000}
        timeout=${3:-30}
        check_vllm_health $port $timeout
        ;;
    *)
        echo "Usage: $0 {single|multi} [port] [timeout]"
        echo "Examples:"
        echo "  $0 single 8000 30    # Check single instance"
        echo "  $0 multi             # Check multiple instances"
        ;;
esac