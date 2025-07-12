#!/bin/bash

echo "🔥 Warming up Qwen2.5 32B model..."

PORT=${1:-8000}

# Check if vLLM is running
if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "❌ vLLM not running on port $PORT"
    echo "Start it first with: ./deploy_vllm.sh start"
    exit 1
fi

# Series of warm-up requests with increasing complexity
warmup_requests=(
    '{"model": "qwen2.5-32b", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1}'
    '{"model": "qwen2.5-32b", "messages": [{"role": "user", "content": "Hello, how are you?"}], "max_tokens": 10}'
    '{"model": "qwen2.5-32b", "messages": [{"role": "user", "content": "Explain quantum computing briefly"}], "max_tokens": 50}'
    '{"model": "qwen2.5-32b", "messages": [{"role": "user", "content": "Write a short story about AI"}], "max_tokens": 100}'
    '{"model": "qwen2.5-32b", "messages": [{"role": "user", "content": "Analyze the benefits of renewable energy sources and their impact on climate change"}], "max_tokens": 150}'
)

total_warmup_time=0

for i in "${!warmup_requests[@]}"; do
    echo "Warmup request $((i+1))/${#warmup_requests[@]}..."
    
    start_time=$(date +%s.%N)
    response=$(curl -s -X POST "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "${warmup_requests[$i]}")
    end_time=$(date +%s.%N)
    
    duration=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "0")
    total_warmup_time=$(echo "$total_warmup_time + $duration" | bc -l 2>/dev/null || echo "0")
    
    # Check if request was successful
    if echo "$response" | grep -q "choices"; then
        echo "✅ Request took: ${duration}s"
    else
        echo "❌ Request failed"
        echo "Response: $response"
    fi
    
    sleep 1
done

echo ""
echo "✅ Model warmup completed!"
echo "📊 Total warmup time: ${total_warmup_time}s"
echo "🚀 Model is now ready for optimal performance"

# Final performance test
echo ""
echo "🧪 Running final performance test..."
start_time=$(date +%s.%N)
curl -s -X POST "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model": "qwen2.5-32b", "messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 10}' > /dev/null
end_time=$(date +%s.%N)

final_duration=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "0")
echo "🎯 Warmed-up inference time: ${final_duration}s"