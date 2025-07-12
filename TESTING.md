# vLLM Testing Guide

Comprehensive testing guide for vLLM Qwen2.5 32B deployment on ThunderCompute.

## 🕒 First Load Testing

### Expected Timeline
- **First-time deployment**: 10-25 minutes
- **Subsequent restarts**: 4-10 minutes
- **Model download**: 5-15 minutes (first time only)
- **Model loading**: 3-8 minutes
- **First inference**: 5-15 seconds
- **Normal inference**: 2-5 seconds

### Run First Load Test
```bash
# Make scripts executable
chmod +x *.sh

# Test first-time deployment with timing
./test_first_load.sh
```

## 🧪 Testing Scripts

### 1. First Load Performance Test
```bash
./test_first_load.sh
```
- Measures complete deployment time
- Tests first request latency
- Compares subsequent request speeds
- Provides performance summary

### 2. Health Check
```bash
# Single instance
./quick_health_check.sh single 8000 30

# Multiple environments
./quick_health_check.sh multi
```

### 3. Model Warmup
```bash
# Warm up model for optimal performance
./warmup_model.sh 8000
```

### 4. Memory Monitoring
```bash
# Monitor in real-time (updates every 5 seconds)
./memory_monitor.sh 8000 5

# Monitor with different interval
./memory_monitor.sh 8000 10
```

## 💾 Memory Management

### Model Persistence
- **GPU Memory**: Model stays loaded until service stops
- **Disk Cache**: Model files cached permanently (~60GB)
- **Auto-restart**: Service starts automatically on boot

### Memory Commands
```bash
# Check GPU memory usage
nvidia-smi

# Check model cache size
du -sh ~/.cache/huggingface/hub/

# Monitor service status
sudo systemctl status vllm-qwen.service

# View memory usage over time
./memory_monitor.sh
```

## 🌍 Multi-Environment Setup

### Environment Configurations
- **Development**: `configs/dev.env` (Port 8000, reduced memory)
- **Staging**: `configs/staging.env` (Port 8001, balanced)
- **Production**: `configs/prod.env` (Port 8000, optimized)

### Deploy with Environment
```bash
# Load development config
source configs/dev.env && ./deploy_vllm.sh all

# Load production config  
source configs/prod.env && ./deploy_vllm.sh all

# Load staging config
source configs/staging.env && ./deploy_vllm.sh all
```

### Test Multiple Environments
```bash
# Check all environments
./quick_health_check.sh multi

# Individual environment checks
./quick_health_check.sh single 8000  # Production
./quick_health_check.sh single 8001  # Staging
```

## 📊 Performance Benchmarks

### Expected Performance (A100 80GB)
- **Throughput**: 400-600 tokens/s (single request)
- **Concurrency**: Optimal at 1-3 concurrent requests
- **Memory**: ~65-70GB GPU memory usage
- **First Token**: 2-5 seconds
- **Subsequent Tokens**: 50-100ms each

### Run Benchmarks
```bash
# Built-in performance test
./deploy_vllm.sh test

# Custom benchmark with monitoring
./memory_monitor.sh 8000 2 &
./deploy_vllm.sh test
killall memory_monitor.sh
```

## 🔍 Troubleshooting Tests

### Common Issues & Tests

**1. Out of Memory**
```bash
# Test with reduced memory
sed -i 's/GPU_MEMORY_UTIL=0.85/GPU_MEMORY_UTIL=0.75/' deploy_vllm.sh
./deploy_vllm.sh restart
```

**2. Slow Response Times**
```bash
# Warm up model
./warmup_model.sh

# Check GPU utilization
watch nvidia-smi
```

**3. Service Issues**
```bash
# Check service status
./deploy_vllm.sh status

# View detailed logs
./deploy_vllm.sh logs

# Manual restart
./deploy_vllm.sh restart
```

## 📋 Pre-Production Checklist

### Before Going Live
- [ ] Run first load test: `./test_first_load.sh`
- [ ] Verify memory usage: `./memory_monitor.sh`
- [ ] Test health checks: `./quick_health_check.sh`
- [ ] Warm up model: `./warmup_model.sh`
- [ ] Run performance tests: `./deploy_vllm.sh test`
- [ ] Test auto-restart: `sudo reboot` (check service starts)
- [ ] Verify multi-environment: `./quick_health_check.sh multi`

### Performance Validation
```bash
# Complete validation sequence
./test_first_load.sh
./warmup_model.sh
./deploy_vllm.sh test
./memory_monitor.sh 8000 5
```

## 🚀 Production Tips

1. **Always warm up** after restart: `./warmup_model.sh`
2. **Monitor continuously**: `./memory_monitor.sh &`
3. **Use environment configs** for different deployments
4. **Test health regularly**: `./quick_health_check.sh`
5. **Keep logs**: Service logs automatically via journald

## 📈 Optimization Guidelines

### For Different Use Cases

**High Throughput (Multiple Users)**
```bash
# Reduce context length, increase memory efficiency
MAX_MODEL_LEN=4096
GPU_MEMORY_UTIL=0.90
```

**Low Latency (Single User)**
```bash
# Increase context, optimize for response time
MAX_MODEL_LEN=8192
GPU_MEMORY_UTIL=0.80
./warmup_model.sh  # Always warm up
```

**Development/Testing**
```bash
# Use development config
source configs/dev.env
MAX_MODEL_LEN=2048
GPU_MEMORY_UTIL=0.70
```