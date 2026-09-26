# Scalable LLM Inference Platform

A production-grade inference platform for quantized large language models (LLMs), demonstrating infrastructure skills in serving, scaling, observability, and cost optimization.

## Overview

This project deploys a quantized LLM (Mistral-7B or Llama-3-8B in INT4) behind a FastAPI gateway with observability (Prometheus + Grafana), orchestrated via Docker Compose, and load-tested to measure real-world performance.

**Target Deployment:** Single GPU (~8GB VRAM) with concurrent request handling, sub-500ms p99 latency, and ~100+ tokens/sec throughput.

## Quick Start

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (for GPU inference)
- Docker and Docker Compose (optional, for full stack)

### Setup

```bash
# Clone and navigate
git clone <repo-url>
cd llm-inference-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download and test the model
python src/serving/download_model.py

# Run the API server
python src/gateway/app.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Client Requests                        │
└────────────────┬────────────────────────────────────────────┘
                 │
        ┌────────▼────────┐
        │   FastAPI       │
        │   Gateway       │ (Rate limiting, API-key auth,
        │                 │  async queue)
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │   vLLM Server   │ (Continuous batching,
        │                 │  PagedAttention)
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │ Mistral-7B INT4 │
        │  (quantized)    │
        └─────────────────┘

        Observability: Prometheus → Grafana
```

## Benchmark Results

Measured on a rented NVIDIA L4 (24GB), AWQ-quantized Mistral-7B served via
vLLM. Full methodology, all 11 experiments, and how each number was derived:
[`benchmarks/EXPERIMENTS.md`](benchmarks/EXPERIMENTS.md).

| Metric | Value |
|--------|-------|
| Throughput (10 concurrent users) | ~308 tokens/sec (vs 37.54 tok/s naive sequential baseline — ~8x) |
| p50 Latency (10 concurrent users) | 1580 ms |
| p99 Latency (10 concurrent users) | 1910 ms |
| Time-to-first-token (streaming, avg) | 78.8 ms (vs 1497 ms non-streaming — ~19x) |
| Concurrent Users Supported | ~256-300 (vLLM's own scheduler limit, confirmed via its internal metrics) |
| GPU Memory Usage | ~20.4 GB / 23 GB (L4) |

![Grafana dashboard showing live gateway and k6 load test metrics during the concurrency stress tests](monitoring/grafana-dashboards/dashboard-screenshots.png)

## Project Structure

```
llm-inference-platform/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git exclusions
├── src/
│   ├── gateway/              # FastAPI app
│   │   ├── app.py
│   │   ├── auth.py
│   │   └── queue.py
│   └── serving/              # vLLM / model logic
│       ├── model_loader.py
│       └── download_model.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
├── loadtest/                 # Locust/k6 scripts
│   ├── locustfile.py
│   └── results/
└── benchmarks/               # Analysis & reports
    └── results.md
```

## Development Progress

### Step 1: Model Quantization
- [x] Download Mistral-7B-Instruct INT4 (AWQ pre-quantized, for vLLM; bitsandbytes on-the-fly NF4 for the baseline path)
- [x] Verify model loads and generates text
- [x] Benchmark baseline tokens/sec — 37.54 tok/s

### Step 2: vLLM Integration
- [x] Add a vLLM server launcher for CUDA/Linux hosts
- [x] Add a sequential client benchmark for the vLLM endpoint
- [x] Add a baseline-vs-vLLM comparison script
- [x] Replace naive `generate()` with vLLM server
- [x] Measure throughput improvement — 59.74 tok/s sequential (1.59x), ~308 tok/s at 10 concurrent users (~8x)

### Step 3: FastAPI Gateway
- [x] Request validation (Pydantic) and async proxy to vLLM
- [x] API-key authentication
- [x] Rate limiting
- [x] Async queue / backpressure handling
- [x] Streaming responses (SSE) — ~19x better time-to-first-token vs non-streaming

### Step 4: Containerization
- [x] Dockerfile for model + API
- [x] Docker Compose orchestration

### Step 5: Observability
- [x] Prometheus metrics (latency, throughput, queue depth)
- [x] Grafana dashboard

### Step 6: Load Testing & Tuning
- [x] k6 load tests — fixed-load and staircase stress tests up to 1000 concurrent users
- [x] p50/p95/p99 latency benchmarks
- [x] Batch size / scheduler tuning — found and confirmed vLLM's real concurrency ceiling (`max-num-seqs=256`)

### Step 7: Polish & Publish
- [ ] Final README, commit history
- [ ] Push to GitHub
- [ ] CI workflow (optional)

## Next Steps

- [ ] Benchmark against other inference frameworks (llama.cpp, TensorRT)
- [ ] Add Kubernetes manifests for autoscaling
- [ ] Quantize other models (Llama-3, Mixtral)
- [ ] Fine-tuning pipeline integration
