#!/bin/bash

echo "=== vLLM First Load Performance Test ==="
echo "Start time: $(date)"
START_TIME=$(date +%s)

# Check if model is already loaded
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "⚠️ vLLM already running - stopping for clean test"
    ./deploy_vllm.sh stop
    sleep 5
fi

echo -e "\n🚀 Starting fresh deployment..."
DEPLOY_START=$(date +%s)

# Start deployment
./deploy_vllm.sh start

DEPLOY_END=$(date +%s)
DEPLOY_TIME=$((DEPLOY_END - DEPLOY_START))

echo -e "\n⏱️ Deployment completed in: ${DEPLOY_TIME} seconds"

# Test first request
echo -e "\n🧪 Testing first request..."
FIRST_REQ_START=$(date +%s)

RESPONSE=$(curl -s -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-32b",
    "messages": [{"role": "user", "content": "Say hello in exactly 5 words"}],
    "max_tokens": 10,
    "temperature": 0.1
  }')

FIRST_REQ_END=$(date +%s)
FIRST_REQ_TIME=$((FIRST_REQ_END - FIRST_REQ_START))

echo "First request completed in: ${FIRST_REQ_TIME} seconds"

# Test subsequent requests (should be faster)
echo -e "\n🔄 Testing subsequent request speed..."
for i in {1..3}; do
    REQ_START=$(date +%s)
    curl -s -X POST "http://localhost:8000/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "qwen2.5-32b",
        "messages": [{"role": "user", "content": "Count to 3"}],
        "max_tokens": 10,
        "temperature": 0.1
      }' > /dev/null
    REQ_END=$(date +%s)
    REQ_TIME=$((REQ_END - REQ_START))
    echo "Request $i: ${REQ_TIME} seconds"
done

TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - START_TIME))

echo -e "\n📊 Performance Summary:"
echo "========================"
echo "Total test time: ${TOTAL_TIME} seconds"
echo "Deployment time: ${DEPLOY_TIME} seconds" 
echo "First request: ${FIRST_REQ_TIME} seconds"
echo "GPU memory used: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) MB"
echo "Service status: $(systemctl is-active vllm-qwen.service)"
echo "End time: $(date)"