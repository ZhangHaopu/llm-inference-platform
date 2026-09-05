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

*To be updated after load testing*

| Metric | Value |
|--------|-------|
| Throughput (tokens/sec) | -- |
| p50 Latency (ms) | -- |
| p99 Latency (ms) | -- |
| Concurrent Users Supported | -- |
| Memory Usage (GB) | -- |

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
- [ ] Download Mistral-7B-Instruct INT4 (pre-quantized from HuggingFace)
- [ ] Verify model loads and generates text
- [ ] Benchmark baseline tokens/sec

### Step 2: vLLM Integration
- [x] Add a vLLM server launcher for CUDA/Linux hosts
- [x] Add a sequential client benchmark for the vLLM endpoint
- [x] Add a baseline-vs-vLLM comparison script
- [ ] Replace naive `generate()` with vLLM server
- [ ] Measure throughput improvement (delta from Step 1) — needs a GPU run

### Step 3: FastAPI Gateway
- [x] Request validation (Pydantic) and async proxy to vLLM
- [x] API-key authentication
- [ ] Rate limiting
- [ ] Async queue / backpressure handling

### Step 4: Containerization
- [ ] Dockerfile for model + API
- [ ] Docker Compose orchestration

### Step 5: Observability
- [ ] Prometheus metrics (latency, throughput, queue depth)
- [ ] Grafana dashboard

### Step 6: Load Testing & Tuning
- [ ] Locust/k6 load tests
- [ ] p50/p95/p99 latency benchmarks
- [ ] Batch size optimization

### Step 7: Polish & Publish
- [ ] Final README, commit history
- [ ] Push to GitHub
- [ ] CI workflow (optional)

## Next Steps

- [ ] Benchmark against other inference frameworks (llama.cpp, TensorRT)
- [ ] Add Kubernetes manifests for autoscaling
- [ ] Quantize other models (Llama-3, Mixtral)
- [ ] Fine-tuning pipeline integration

## Notes

This project is intentionally scope-disciplined: a complete, well-documented v1 with real load-test numbers beats a half-finished "production-ready" version. See `.claude/01_initial.md` for design philosophy and budget notes.
