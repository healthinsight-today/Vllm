# vLLM Qwen2.5 32B Deployment for ThunderCompute

Production-ready deployment script for running Qwen2.5 32B with vLLM on ThunderCompute A100 80GB instances.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/vllm-qwen-thundercompute.git
cd vllm-qwen-thundercompute

# Make script executable
chmod +x deploy_vllm.sh

# Run complete deployment
./deploy_vllm.sh all
```

## 📋 Prerequisites

- **GPU**: NVIDIA A100 80GB (minimum 75GB VRAM for Qwen2.5 32B)
- **OS**: Ubuntu 20.04+ or similar Linux distribution
- **CUDA**: Version 12.1+ 
- **Python**: 3.8+
- **RAM**: 32GB+ system memory recommended
- **Storage**: 100GB+ free space for model weights

## 🛠 Installation Options

### Option 1: Complete Setup (Recommended)
```bash
./deploy_vllm.sh all
```
Runs the full pipeline: system check → install → service setup → start → test

### Option 2: Step-by-Step
```bash
# Install dependencies and setup
./deploy_vllm.sh install

# Start the service
./deploy_vllm.sh start

# Run tests
./deploy_vllm.sh test
```

## 🎯 Key Features

- **Systemd Service**: Production-ready service management with auto-restart
- **Optimized Configuration**: Settings tuned specifically for Qwen2.5 32B on A100 80GB
- **Comprehensive Testing**: Functionality and performance benchmarks
- **Resource Monitoring**: GPU, CPU, and memory usage tracking
- **Error Handling**: Robust error detection and recovery

## 📊 Performance Expectations

Based on vLLM benchmarks for 32B models on A100 80GB:

- **Throughput**: ~400-600 tokens/s (single request)
- **Latency**: 2-5 seconds for first token
- **Concurrency**: Optimal at 1-3 concurrent requests
- **Memory Usage**: ~65-70GB GPU memory

## 🔧 Configuration

Key parameters optimized for Qwen2.5 32B:

```bash
MODEL_NAME="Qwen/Qwen2.5-32B-Instruct"
VLLM_PORT=8000
MAX_MODEL_LEN=8192
GPU_MEMORY_UTIL=0.85  # Prevents OOM with 32B model
```

## 📝 Usage Commands

| Command | Description |
|---------|-------------|
| `./deploy_vllm.sh install` | Install all dependencies and create service |
| `./deploy_vllm.sh start` | Start vLLM server |
| `./deploy_vllm.sh stop` | Stop vLLM server |
| `./deploy_vllm.sh restart` | Restart vLLM server |
| `./deploy_vllm.sh test` | Run functionality and performance tests |
| `./deploy_vllm.sh health` | Check server health |
| `./deploy_vllm.sh logs` | Show recent logs |
| `./deploy_vllm.sh status` | Show detailed service status |
| `./deploy_vllm.sh monitor` | Show system resource usage |

## 🌐 API Usage

Once deployed, the server provides OpenAI-compatible endpoints:

```bash
# Test basic completion
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-32b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Python Client Example

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # vLLM doesn't require API key by default
)

response = client.chat.completions.create(
    model="qwen2.5-32b",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    max_tokens=200
)

print(response.choices[0].message.content)
```

## 🔍 Monitoring & Debugging

### Check Service Status
```bash
sudo systemctl status vllm-qwen.service
```

### View Live Logs
```bash
sudo journalctl -f -u vllm-qwen.service
```

### Monitor GPU Usage
```bash
watch nvidia-smi
```

### Performance Monitoring
```bash
./deploy_vllm.sh monitor
```

## ⚡ Performance Tuning

For different use cases, adjust these parameters:

### High Throughput (Multiple Users)
```bash
# Reduce max_model_len to fit more requests in memory
MAX_MODEL_LEN=4096
GPU_MEMORY_UTIL=0.90
```

### Low Latency (Single User)
```bash
# Increase context length for longer conversations
MAX_MODEL_LEN=16384
GPU_MEMORY_UTIL=0.80
```

## 🚨 Troubleshooting

### Common Issues

**1. Out of Memory Errors**
```bash
# Reduce GPU memory utilization
GPU_MEMORY_UTIL=0.75
```

**2. Slow First Token**
```bash
# Model loading takes 5-10 minutes on first start
# Monitor with: sudo journalctl -f -u vllm-qwen.service
```

**3. Service Won't Start**
```bash
# Check logs
sudo journalctl -u vllm-qwen.service --no-pager -l

# Verify GPU availability
nvidia-smi
```

**4. Port Already in Use**
```bash
# Kill processes on port 8000
sudo lsof -ti:8000 | xargs kill -9
```

## 💰 Cost Optimization (ThunderCompute)

At $0.78/hour for A100 80GB:

- **Development**: Use `./deploy_vllm.sh stop` when not in use
- **Production**: Keep running for consistent response times
- **Testing**: Use smaller models (7B/14B) for development

## 🔒 Security Considerations

- **Firewall**: Restrict port 8000 access as needed
- **API Key**: Add authentication for production use
- **Resource Limits**: Service includes built-in resource constraints

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test on ThunderCompute A100 80GB
4. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: GitHub Issues for bugs and feature requests
- **Discussions**: GitHub Discussions for questions
- **vLLM Documentation**: [vLLM Docs](https://docs.vllm.ai/)
- **ThunderCompute**: [ThunderCompute Platform](https://www.thundercompute.com/)

## 🏷 Tags

`vllm` `qwen2.5` `llm-serving` `gpu-cloud` `thundercompute` `a100` `inference` `production`

---

**⚠️ Note**: This script is optimized specifically for Qwen2.5 32B on A100 80GB. For other models or hardware, adjust the configuration parameters accordingly.