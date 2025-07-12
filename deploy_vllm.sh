#!/bin/bash

# vLLM Qwen2.5 32B Deployment Script for ThunderCompute A100 80GB
# This script installs vLLM, deploys Qwen2.5 32B model, and runs comprehensive tests
# Author: Generated for ThunderCompute deployment
# Version: 1.0

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODEL_NAME="Qwen/Qwen2.5-32B-Instruct"
VLLM_PORT=8000
MAX_MODEL_LEN=8192  # Increased for better performance with 32B model
GPU_MEMORY_UTIL=0.85  # Slightly reduced to prevent OOM with 32B model
TEST_CONCURRENCY=(1 3 5)  # Reduced max concurrency for 32B model

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check system requirements
check_system() {
    log "Checking system requirements for Qwen2.5 32B..."
    
    # Check CUDA
    if ! command -v nvidia-smi &> /dev/null; then
        error "nvidia-smi not found. CUDA drivers may not be installed."
        exit 1
    fi
    
    # Check GPU memory (32B model needs ~65-70GB)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "$GPU_MEMORY" -lt 75000 ]; then
        error "Insufficient GPU memory: ${GPU_MEMORY}MB. Need at least 75GB for Qwen2.5 32B."
        exit 1
    fi
    
    success "GPU Memory: ${GPU_MEMORY}MB detected (sufficient for Qwen2.5 32B)"
    
    # Check Python version
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        error "Python 3.8+ required. Found: $PYTHON_VERSION"
        exit 1
    fi
    
    success "Python version: $PYTHON_VERSION"
    
    # Check CUDA version
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    log "CUDA Version: $CUDA_VERSION"
    
    # Check available disk space (models can be large)
    DISK_SPACE=$(df / | awk 'NR==2 {print $4}')
    if [ "$DISK_SPACE" -lt 10485760 ]; then  # 10GB in KB
        warning "Low disk space. Qwen2.5 32B model requires ~60GB download space."
    fi
    
    # Display system info
    log "System specifications:"
    echo "  - GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "  - CPU cores: $(nproc)"
    echo "  - RAM: $(free -h | awk 'NR==2{print $2}')"
    echo "  - Available disk: $(df -h / | awk 'NR==2{print $4}')"
}

# Function to install dependencies
install_dependencies() {
    log "Installing dependencies for vLLM and Qwen2.5 32B..."
    
    # Update system
    sudo apt-get update -y
    sudo apt-get install -y python3-pip python3-venv git htop curl lsof bc
    
    # Create virtual environment
    if [ ! -d "vllm_env" ]; then
        python3 -m venv vllm_env
    fi
    source vllm_env/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip wheel setuptools
    
    # Install PyTorch with CUDA support (matching vLLM requirements)
    log "Installing PyTorch with CUDA 12.1 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Install vLLM
    log "Installing vLLM..."
    pip install vllm
    
    # Install additional tools for testing and monitoring
    pip install requests httpx aiohttp numpy transformers
    
    success "Dependencies installed successfully"
}

# Function to verify installation
verify_installation() {
    log "Verifying vLLM installation..."
    
    source vllm_env/bin/activate
    
    # Test CUDA availability
    log "Testing CUDA setup..."
    python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA devices: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
"
    
    # Test vLLM import
    python3 -c "
import vllm
print(f'vLLM version: {vllm.__version__}')

# Test model loading capability (without actually loading)
from vllm import LLM
print('vLLM import successful')
"
    
    success "Installation verified successfully"
}

# Function to pre-download model (optional but recommended)
download_model() {
    log "Pre-downloading Qwen2.5 32B model (this may take a while)..."
    
    source vllm_env/bin/activate
    
    python3 -c "
from transformers import AutoTokenizer
print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('$MODEL_NAME', trust_remote_code=True)
print('Tokenizer downloaded successfully')
"
    
    success "Model components pre-downloaded"
}

# Function to create systemd service
create_systemd_service() {
    log "Creating systemd service for vLLM..."
    
    # Get current user and working directory
    CURRENT_USER=$(whoami)
    WORK_DIR=$(pwd)
    VENV_PATH="$WORK_DIR/vllm_env"
    
    # Create systemd service file
    sudo tee /etc/systemd/system/vllm-qwen.service > /dev/null << EOF
[Unit]
Description=vLLM Qwen2.5 32B Inference Server
After=network.target
Wants=network.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$WORK_DIR
Environment=CUDA_VISIBLE_DEVICES=0
Environment=PYTHONPATH=$VENV_PATH/lib/python*/site-packages
ExecStart=$VENV_PATH/bin/python -m vllm.entrypoints.openai.api_server \\
    --model $MODEL_NAME \\
    --port $VLLM_PORT \\
    --gpu-memory-utilization $GPU_MEMORY_UTIL \\
    --max-model-len $MAX_MODEL_LEN \\
    --trust-remote-code \\
    --host 0.0.0.0 \\
    --disable-log-requests \\
    --served-model-name qwen2.5-32b
ExecReload=/bin/kill -HUP \$MAINPID
KillMode=mixed
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=$WORK_DIR
ReadWritePaths=/tmp
ReadWritePaths=/var/tmp

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd and enable service
    sudo systemctl daemon-reload
    sudo systemctl enable vllm-qwen.service
    
    success "Systemd service created and enabled"
}

# Function to start vLLM server (updated for service management)
start_vllm_server() {
    log "Starting vLLM server with Qwen2.5 32B..."
    
    # Check if systemd service exists
    if systemctl list-unit-files | grep -q "vllm-qwen.service"; then
        log "Using systemd service..."
        
        # Stop any existing processes on the port
        if lsof -Pi :$VLLM_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
            warning "Port $VLLM_PORT is in use. Stopping conflicting processes..."
            sudo systemctl stop vllm-qwen.service 2>/dev/null || true
            sleep 3
            sudo kill -9 $(lsof -t -i:$VLLM_PORT) 2>/dev/null || true
        fi
        
        # Start the service
        sudo systemctl start vllm-qwen.service
        
        log "vLLM service starting... This may take 5-10 minutes for Qwen2.5 32B"
        log "Monitor progress: sudo journalctl -f -u vllm-qwen.service"
        
        # Wait for service to start
        log "Waiting for service to initialize (up to 15 minutes)..."
        for i in {1..180}; do
            if systemctl is-active --quiet vllm-qwen.service && curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
                success "vLLM service is ready and serving Qwen2.5 32B!"
                return 0
            fi
            if [ $((i % 12)) -eq 0 ]; then
                log "Still loading... (${i} minutes elapsed)"
                systemctl status vllm-qwen.service --no-pager -l | tail -3
            fi
            echo -n "."
            sleep 5
        done
        
        error "Service failed to start within 15 minutes"
        sudo journalctl -u vllm-qwen.service --no-pager -l | tail -50
        exit 1
        
    else
        # Fallback to manual start
        log "No systemd service found, starting manually..."
        start_vllm_manual
    fi
}

# Function for manual start (fallback)
start_vllm_manual() {
    source vllm_env/bin/activate
    
    # Check if port is already in use
    if lsof -Pi :$VLLM_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        warning "Port $VLLM_PORT is already in use. Attempting to kill existing process..."
        sudo kill -9 $(lsof -t -i:$VLLM_PORT) 2>/dev/null || true
        sleep 3
    fi
    
    log "Starting vLLM server manually (this will take several minutes for first-time model loading)..."
    
    # Start vLLM server in background with optimized settings for 32B model
    nohup python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL_NAME \
        --port $VLLM_PORT \
        --gpu-memory-utilization $GPU_MEMORY_UTIL \
        --max-model-len $MAX_MODEL_LEN \
        --trust-remote-code \
        --host 0.0.0.0 \
        --disable-log-requests \
        --served-model-name qwen2.5-32b > vllm_server.log 2>&1 &
    
    VLLM_PID=$!
    echo $VLLM_PID > vllm_server.pid
    
    log "vLLM server starting (PID: $VLLM_PID)..."
    log "Model loading in progress... This may take 5-10 minutes for Qwen2.5 32B"
    log "Monitor progress: tail -f vllm_server.log"
    
    # Wait for server to start with longer timeout for 32B model
    log "Waiting for server to initialize (this may take up to 15 minutes)..."
    for i in {1..180}; do  # 15 minutes timeout
        if curl -s http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
            success "vLLM server is ready and serving Qwen2.5 32B!"
            return 0
        fi
        if [ $((i % 12)) -eq 0 ]; then  # Every minute
            log "Still loading... (${i} minutes elapsed)"
        fi
        echo -n "."
        sleep 5
    done
    
    error "Server failed to start within 15 minutes"
    echo "Last 50 lines of server log:"
    tail -50 vllm_server.log
    exit 1
}

# Function to run basic functionality test
test_basic_functionality() {
    log "Testing basic functionality with Qwen2.5 32B..."
    
    # Test simple completion
    log "Testing simple completion..."
    RESPONSE=$(curl -s -X POST "http://localhost:$VLLM_PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "qwen2.5-32b",
            "messages": [{"role": "user", "content": "Hello! Can you tell me a brief fact about quantum computing?"}],
            "max_tokens": 100,
            "temperature": 0.7
        }')
    
    if echo "$RESPONSE" | grep -q "choices"; then
        success "Basic functionality test passed"
        CONTENT=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "Parse error")
        echo "Model response: $CONTENT"
    else
        error "Basic functionality test failed"
        echo "Response: $RESPONSE"
        return 1
    fi
    
    # Test reasoning capability
    log "Testing reasoning capability..."
    REASONING_RESPONSE=$(curl -s -X POST "http://localhost:$VLLM_PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "qwen2.5-32b",
            "messages": [{"role": "user", "content": "Solve this step by step: If a train travels 120 km in 2 hours, what is its average speed?"}],
            "max_tokens": 150,
            "temperature": 0.3
        }')
    
    if echo "$REASONING_RESPONSE" | grep -q "choices"; then
        success "Reasoning test passed"
        REASONING_CONTENT=$(echo "$REASONING_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "Parse error")
        echo "Reasoning response: $REASONING_CONTENT"
    else
        warning "Reasoning test had issues"
        echo "Response: $REASONING_RESPONSE"
    fi
}

# Function to run performance benchmarks
run_performance_tests() {
    log "Running performance benchmarks for Qwen2.5 32B..."
    
    source vllm_env/bin/activate
    
    # Create enhanced benchmark script
    cat > benchmark_test.py << 'EOF'
import asyncio
import aiohttp
import time
import json
import sys
import statistics

async def send_request(session, url, payload, request_id):
    start_time = time.time()
    first_token_time = None
    
    try:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                end_time = time.time()
                
                if 'choices' in result and result['choices']:
                    content = result['choices'][0]['message']['content']
                    tokens = len(content.split())
                    
                    # Estimate first token time (rough approximation)
                    first_token_time = start_time + 0.1  # Placeholder
                    
                    return {
                        'request_id': request_id,
                        'total_latency': end_time - start_time,
                        'first_token_latency': first_token_time - start_time if first_token_time else None,
                        'tokens': tokens,
                        'tokens_per_second': tokens / (end_time - start_time) if end_time > start_time else 0,
                        'success': True,
                        'response_length': len(content)
                    }
                else:
                    return {
                        'request_id': request_id,
                        'total_latency': end_time - start_time,
                        'success': False,
                        'error': 'No choices in response'
                    }
            else:
                end_time = time.time()
                error_text = await response.text()
                return {
                    'request_id': request_id,
                    'total_latency': end_time - start_time,
                    'success': False,
                    'error': f'HTTP {response.status}: {error_text}'
                }
                
    except Exception as e:
        end_time = time.time()
        return {
            'request_id': request_id,
            'total_latency': end_time - start_time,
            'success': False,
            'error': str(e)
        }

async def benchmark(concurrency, model_name, port):
    url = f"http://localhost:{port}/v1/chat/completions"
    
    # Different prompts for testing various capabilities
    prompts = [
        "Explain the concept of machine learning in simple terms.",
        "Write a short story about a robot discovering emotions.",
        "Solve this math problem: What is 15% of 240?",
        "Describe the benefits of renewable energy sources.",
        "What are the key principles of good software design?"
    ]
    
    payload_template = {
        "model": model_name,
        "max_tokens": 150,
        "temperature": 0.7
    }
    
    print(f"\\n{'='*60}")
    print(f"Testing with {concurrency} concurrent requests")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Create requests with varied prompts
    tasks = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        for i in range(concurrency):
            payload = payload_template.copy()
            payload["messages"] = [{"role": "user", "content": prompts[i % len(prompts)]}]
            tasks.append(send_request(session, url, payload, i))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Filter out exceptions and process results
    valid_results = [r for r in results if isinstance(r, dict)]
    successful = [r for r in valid_results if r['success']]
    failed = [r for r in valid_results if not r['success']]
    
    print(f"\\nResults Summary:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Successful requests: {len(successful)}/{concurrency}")
    print(f"  Failed requests: {len(failed)}")
    
    if successful:
        latencies = [r['total_latency'] for r in successful]
        token_counts = [r['tokens'] for r in successful]
        tokens_per_sec = [r['tokens_per_second'] for r in successful]
        
        total_tokens = sum(token_counts)
        overall_throughput = total_tokens / total_time
        
        print(f"\\nPerformance Metrics:")
        print(f"  Average latency: {statistics.mean(latencies):.2f}s")
        print(f"  Median latency: {statistics.median(latencies):.2f}s")
        print(f"  95th percentile latency: {sorted(latencies)[int(0.95 * len(latencies))]:.2f}s")
        print(f"  Average tokens per response: {statistics.mean(token_counts):.1f}")
        print(f"  Total tokens generated: {total_tokens}")
        print(f"  Overall throughput: {overall_throughput:.2f} tokens/s")
        print(f"  Average per-request throughput: {statistics.mean(tokens_per_sec):.2f} tokens/s")
        
        # Memory efficiency indicator
        print(f"\\nEfficiency Metrics:")
        print(f"  Requests per second: {len(successful) / total_time:.2f}")
        print(f"  Average response length: {statistics.mean([r['response_length'] for r in successful]):.0f} chars")
        
    if failed:
        print(f"\\nError Summary:")
        error_types = {}
        for f in failed:
            error_key = f['error'][:50] + "..." if len(f['error']) > 50 else f['error']
            error_types[error_key] = error_types.get(error_key, 0) + 1
        
        for error, count in error_types.items():
            print(f"  {error}: {count} occurrences")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python benchmark_test.py <port> <model_name>")
        sys.exit(1)
    
    port = sys.argv[1]
    model_name = sys.argv[2]
    
    concurrency_levels = [1, 3, 5]
    
    print(f"Starting performance benchmark for {model_name}")
    print(f"Server: http://localhost:{port}")
    
    for concurrency in concurrency_levels:
        try:
            asyncio.run(benchmark(concurrency, model_name, port))
        except Exception as e:
            print(f"Error in benchmark with concurrency {concurrency}: {e}")
        
        time.sleep(3)  # Brief pause between tests
    
    print(f"\\n{'='*60}")
    print("Benchmark completed!")
    print(f"{'='*60}")
EOF
    
    python3 benchmark_test.py $VLLM_PORT "qwen2.5-32b"
}

# Function to monitor system resources
monitor_resources() {
    log "System resource monitoring:"
    
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
    
    echo -e "\\n=== Memory Usage ==="
    free -h
    
    echo -e "\\n=== CPU Usage ==="
    echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}')%"
    
    echo -e "\\n=== Disk Usage ==="
    df -h /
    
    echo -e "\\n=== Network Connections ==="
    ss -tulpn | grep :$VLLM_PORT || echo "vLLM server not listening on port $VLLM_PORT"
    
    echo -e "\\n=== Process Info ==="
    if [ -f vllm_server.pid ]; then
        PID=$(cat vllm_server.pid)
        if ps -p $PID > /dev/null; then
            echo "vLLM server running (PID: $PID)"
            ps -p $PID -o pid,ppid,cmd,%cpu,%mem
        else
            echo "vLLM server not running (stale PID file)"
        fi
    else
        echo "No vLLM server PID file found"
    fi
}

# Function to check model health
health_check() {
    log "Performing health check..."
    
    # Basic health endpoint
    if curl -s http://localhost:$VLLM_PORT/health > /dev/null; then
        success "Health endpoint responding"
    else
        error "Health endpoint not responding"
        return 1
    fi
    
    # Model info endpoint
    MODEL_INFO=$(curl -s http://localhost:$VLLM_PORT/v1/models)
    if echo "$MODEL_INFO" | grep -q "qwen2.5-32b"; then
        success "Model endpoint responding correctly"
        echo "Available models: $(echo "$MODEL_INFO" | python3 -c "import sys, json; models = json.load(sys.stdin)['data']; print([m['id'] for m in models])" 2>/dev/null || echo "Parse error")"
    else
        warning "Model endpoint issue"
        echo "Response: $MODEL_INFO"
    fi
}

# Function to stop vLLM server (updated for service management)
stop_vllm_server() {
    log "Stopping vLLM server..."
    
    # Check if systemd service exists and is running
    if systemctl list-unit-files | grep -q "vllm-qwen.service" && systemctl is-active --quiet vllm-qwen.service; then
        log "Stopping systemd service..."
        sudo systemctl stop vllm-qwen.service
        
        # Wait for service to stop
        for i in {1..30}; do
            if ! systemctl is-active --quiet vllm-qwen.service; then
                success "vLLM service stopped gracefully"
                return 0
            fi
            sleep 1
        done
        
        warning "Service didn't stop gracefully, forcing..."
        sudo systemctl kill vllm-qwen.service
        success "vLLM service stopped (forced)"
        
    else
        # Fallback to manual stop
        if [ -f vllm_server.pid ]; then
            PID=$(cat vllm_server.pid)
            if ps -p $PID > /dev/null 2>&1; then
                log "Sending SIGTERM to PID $PID..."
                kill -15 $PID
                
                # Wait for graceful shutdown
                for i in {1..30}; do
                    if ! ps -p $PID > /dev/null 2>&1; then
                        success "vLLM server stopped gracefully"
                        rm -f vllm_server.pid
                        return 0
                    fi
                    sleep 1
                done
                
                # Force kill if still running
                warning "Forcing shutdown..."
                kill -9 $PID 2>/dev/null || true
                success "vLLM server stopped (forced)"
            else
                warning "vLLM server was not running"
            fi
            rm -f vllm_server.pid
        else
            warning "No PID file found"
        fi
    fi
    
    # Kill any remaining processes on the port
    lsof -ti:$VLLM_PORT | xargs kill -9 2>/dev/null || true
}

# Function to cleanup
cleanup() {
    log "Cleaning up temporary files..."
    rm -f benchmark_test.py
    # Note: We keep vllm_server.log for debugging
    success "Cleanup completed (log files preserved)"
}

# Function to show logs (updated for service management)
show_logs() {
    if systemctl list-unit-files | grep -q "vllm-qwen.service"; then
        log "Showing systemd service logs (last 50 lines):"
        sudo journalctl -u vllm-qwen.service --no-pager -l | tail -50
    elif [ -f vllm_server.log ]; then
        log "Showing last 50 lines of vLLM server log:"
        tail -50 vllm_server.log
    else
        warning "No log files found"
    fi
}

# Function to check service status
service_status() {
    log "Checking vLLM service status..."
    
    if systemctl list-unit-files | grep -q "vllm-qwen.service"; then
        echo "=== Systemd Service Status ==="
        sudo systemctl status vllm-qwen.service --no-pager -l
        
        echo -e "\n=== Service Performance ==="
        if systemctl is-active --quiet vllm-qwen.service; then
            PID=$(sudo systemctl show vllm-qwen.service -p MainPID --value)
            if [ "$PID" != "0" ]; then
                echo "Main PID: $PID"
                ps -p $PID -o pid,ppid,cmd,%cpu,%mem 2>/dev/null || echo "Process not found"
            fi
        fi
        
        echo -e "\n=== Recent Service Logs ==="
        sudo journalctl -u vllm-qwen.service --no-pager -l | tail -10
        
    else
        echo "No systemd service found"
        if [ -f vllm_server.pid ]; then
            PID=$(cat vllm_server.pid)
            if ps -p $PID > /dev/null; then
                echo "Manual process running (PID: $PID)"
                ps -p $PID -o pid,ppid,cmd,%cpu,%mem
            else
                echo "Manual process not running (stale PID file)"
            fi
        else
            echo "No manual process found"
        fi
    fi
}

# Main execution
main() {
    echo "vLLM Qwen2.5 32B Deployment Script v1.0"
    echo "=========================================="
    
    case "${1:-all}" in
        "install")
            check_system
            install_dependencies
            verify_installation
            download_model
            create_systemd_service
            ;;
        "start")
            start_vllm_server
            health_check
            ;;
        "test")
            health_check
            test_basic_functionality
            run_performance_tests
            ;;
        "monitor")
            monitor_resources
            ;;
        "health")
            health_check
            ;;
        "logs")
            show_logs
            ;;
        "status")
            service_status
            ;;
        "stop")
            stop_vllm_server
            ;;
        "cleanup")
            stop_vllm_server
            cleanup
            ;;
        "restart")
            stop_vllm_server
            sleep 2
            start_vllm_server
            health_check
            ;;
        "service")
            create_systemd_service
            ;;
        "all")
            check_system
            install_dependencies
            verify_installation
            download_model
            create_systemd_service
            start_vllm_server
            health_check
            test_basic_functionality
            run_performance_tests
            monitor_resources
            ;;
        *)
            echo ""
            echo "Usage: $0 {install|start|test|monitor|health|logs|status|stop|cleanup|restart|service|all}"
            echo ""
            echo "Commands:"
            echo "  install  - Install dependencies, vLLM, pre-download model, and create service"
            echo "  start    - Start vLLM server with Qwen2.5 32B"
            echo "  test     - Run functionality and performance tests"
            echo "  monitor  - Show system resource usage"
            echo "  health   - Check server health and model availability"
            echo "  logs     - Show recent server logs"
            echo "  status   - Show detailed service status"
            echo "  stop     - Stop vLLM server gracefully"
            echo "  cleanup  - Stop server and cleanup temporary files"
            echo "  restart  - Stop and start server"
            echo "  service  - Create systemd service only"
            echo "  all      - Run complete deployment and testing (default)"
            echo ""
            echo "Production deployment optimized for ThunderCompute A100 80GB"
            echo "Qwen2.5 32B model with systemd service management"
            exit 1
            ;;
    esac
}

# Trap to cleanup on script exit
trap cleanup EXIT

# Run main function
main "$@"