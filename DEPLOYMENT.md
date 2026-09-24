# Deployment Runbook — GPU Rental Session

A checklist for the actual GPU deployment: what to set up, what to run, what to
watch, and how to load-test correctly. Written to be followed step by step
during a live session, not read once and forgotten — check boxes as you go.

Architecture reminder: **only vLLM runs on the rented GPU.** The gateway,
Prometheus, and Grafana keep running locally (already built and tested against
stubs); `VLLM_URL` just gets pointed at the pod's public address. This
minimizes both GPU cost and the amount of new surface area to debug on
unfamiliar infrastructure.

---

## 0. Pre-flight (before spending any money)

- [ ] `git status` clean, nothing uncommitted
- [ ] Pushed to GitHub (recommended — lets you `git clone` on the pod instead
      of `scp`, and ties benchmark results to a specific commit)
- [ ] RunPod balance loaded (~$10 is comfortably enough at the price point
      chosen)
- [ ] Decided which GPU to rent (Ada-generation, e.g. RTX 2000 Ada / L4 / RTX
      PRO 4000 — see prior discussion for why)

## 1. Rent + launch the pod

- RunPod → **Pods** → Deploy a Pod (not Serverless)
- Template: plain PyTorch/CUDA base (e.g. "Runpod Pytorch 2.8.0") — not a
  pre-built vLLM template, so you control setup yourself
- Expose **port 8000** (vLLM's port) so it's reachable from your Mac
- **Do not leave it running idle.** Terminate (not just "stop") at the end of
  any session where you're not actively working — billing continues either
  way while it's live.

## 2. Initial setup (on the pod)

```bash
python3 --version   # must be 3.10+ — vLLM itself won't import on 3.9
git clone <your-repo-url>
cd llm-inference-platform
pip install -r requirements.txt
```

Note how long `pip install` and the model download take — this is real
GPU-billed time, not free setup.

## 3. Step 1 — Baseline benchmark

```bash
cd src/serving
python benchmark_baseline.py
```

Produces `benchmarks/baseline.json`. Copy it back to your Mac (`scp`) — this
is one half of the Step 2 throughput-delta comparison.

## 4. Step 2 — Launch vLLM, benchmark it

```bash
python vllm_server.py
```

Watch the startup log for:
- Successful model load (no crash, no fallback-to-CPU)
- Confirmation it's actually using the AWQ quantization path, not silently
  ignoring `--quantization` and loading fp16
- `Uvicorn running on 0.0.0.0:8000`

In a second terminal on the pod:

```bash
python benchmark_vllm.py --url http://localhost:8000/v1/completions
```

Copy `vllm.json` back, then locally:

```bash
python src/serving/compare_benchmarks.py
```

→ this is your Step 2 "Nx throughput" number.

## 5. Point the local stack at the pod

On your Mac:

```bash
export VLLM_URL="http://<pod-public-ip>:<mapped-port>"
export API_KEYS="k1,k2,k3,k4,k5,k6,k7,k8,k9,k10"
docker compose -f docker/docker-compose.yml up -d --build --no-deps gateway prometheus grafana
curl http://localhost:8080/health
```

Then send one real completion through the gateway (not `benchmark_vllm.py`
directly) to confirm the full proxy path — auth, rate limit, backpressure,
metrics — works end to end against the *real* model before running any load
test.

## 6. What to watch during testing

- **On the pod**, a spare terminal: `watch -n1 nvidia-smi` — GPU utilization %
  and memory used vs. total. This is what tells you *why* a tuning change
  helped or didn't (compute-bound vs. memory-bound vs. not actually
  bottlenecked on the GPU at all).
- **vLLM's own terminal output** — it logs per-step stats: running/pending
  sequence counts, KV cache usage %. Directly tells you if you're
  memory-constrained.
- **Locally, Grafana** (`localhost:3000`) — request rate, p99 latency,
  `gateway_in_flight_requests` (queue depth), live during the run.
- **k6's own terminal output** — real-time VU count, check pass rate,
  threshold pass/fail as it runs.

## 7. Load testing strategy

Three different tests, answering three different questions — don't conflate
them:

### A. Smoke test — "does the path even work?"
1-2 VUs, ~15-30s, no thresholds worth enforcing yet. **Always run this first**,
every time you change something — it's the cheapest possible way to catch a
broken URL/key/model-not-loaded before burning real time on a full test.

### B. Load test — "what's normal latency/throughput at realistic concurrency?"
This is what `load_test.js` already does. Parameter choices and why:
- **VU count**: 10-20 is a defensible, explainable concurrency for a
  single-GPU portfolio deployment — no need for hundreds.
- **Shape**: ramp up → hold plateau → ramp down. **Only the hold portion is
  what you report** — ramp periods are transient, not steady-state, and
  including them skews percentiles.
- **Hold duration**: needs enough completed requests for percentiles to be
  meaningful, not noisy — rule of thumb, aim for several hundred to 1000+
  requests during the hold. At ~1 req/sec/VU with 15 VUs, 60-90s of hold gets
  you ~900-1350 requests, which is reasonable.
- This produces your **headline number**: p50/p95/p99 latency + aggregate
  throughput at N concurrent users.

### C. Stress / breaking-point test — "how far can this actually go?"
**Nothing built so far measures this**, and it's what the README's "Concurrent
Users Supported" row actually needs. Different shape from B:
- **Staircase**, not a single fixed target: step VU count up at a fixed
  interval (e.g. +5 VUs every 60s) instead of ramping once to one number.
- Keep stepping until *any* of: p99 crosses a threshold you define (e.g.
  >1s), error rate climbs meaningfully (e.g. >5% non-200), or
  `gateway_in_flight_requests` keeps climbing even as latency degrades (a
  clear sign you're past capacity, visible live in Grafana).
- The **last step before degradation** is your real "concurrent users
  supported" number — not a guess, a measured one.
- This needs a second k6 script (a staircase `stages` config) — worth writing
  once you're actually on the pod and have a real baseline to calibrate step
  size against, rather than guessing blind now.

### Methodology details that make results actually valid
- **Warm-up**: send a few throwaway requests immediately after vLLM starts,
  before any measured test — first requests include CUDA graph
  capture/kernel-compile overhead that isn't representative of steady state.
- **Keep prompt/response shape identical across every run** (same prompt
  text, same `max_tokens`) — only vary vLLM's flags between tuning
  iterations, never the test payload, or comparisons stop being apples-to-apples.
- **Per-VU distinct API keys** (already built into `load_test.js`) — without
  this you're measuring the rate limiter, not the system, as the first k6 run
  demonstrated.
- **One flag at a time** during tuning — change one vLLM launch flag, re-run
  test B, compare, keep-or-revert, move to the next. Changing several at once
  makes it impossible to attribute which change caused what.

## 8. Tuning loop

1. Baseline: default flags, run **test B**, record the numbers.
2. Change **one** flag (see list below), restart vLLM (Ctrl+C, relaunch),
   run **test B** again, compare.
3. Keep it if it helped, revert if not. Move to the next flag.
4. After 3-5 iterations, run **test C** once on the best configuration found
   — that's your final "concurrent users supported" figure.

Flags to try, roughly in order of expected impact:
- `--gpu-memory-utilization` (0.90 → 0.95) — more KV cache headroom
- `--max-model-len` (4096 → 2048, if prompts don't need more) — smaller
  per-sequence KV cache footprint, more sequences fit
- `--max-num-seqs` — **not yet exposed in `vllm_server.py`**, needs adding
  (same pattern as the existing flags) or passing raw to `vllm serve`
  directly for a one-off test
- `--enable-chunked-prefill` — **also not yet exposed**, same note
- Gateway's `MAX_CONCURRENT_REQUESTS` — confirm this isn't capping you below
  what vLLM can actually sustain

## 9. Wrap-up

- [ ] Screenshot the Grafana dashboard showing real traffic
- [ ] Save `baseline.json`, `vllm.json`, `compare_benchmarks.py` output, and
      both k6 summaries (B and C) into `benchmarks/`
- [ ] Fill in README's Benchmark Results table with real numbers
- [ ] Copy anything else needed off the pod
- [ ] **Terminate the pod**
- [ ] Commit + push (you do this)
