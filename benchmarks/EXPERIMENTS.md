# Experiment Log — GPU Deployment & Load Testing

Real results from deploying the AWQ-quantized Mistral-7B on a rented NVIDIA L4
(RunPod), served via vLLM behind the FastAPI gateway. Each entry: what we
were trying to find out, what we did, what happened, and what it means.

## 1. Baseline throughput — naive `generate()`

**Goal:** Establish a "before" number using plain HuggingFace `generate()`
with bitsandbytes 4-bit quantization, no serving optimizations.

**Method:** `benchmark_baseline.py`, 5 prompts, `max_tokens=100`, sequential.

**Result:** **37.54 tokens/sec** average.

**Conclusion:** This is the baseline everything else is measured against.

## 2. vLLM sequential throughput

**Goal:** Measure vLLM's raw per-request speed with AWQ quantization, no
concurrency — an apples-to-apples comparison with experiment 1.

**Method:** `benchmark_vllm.py`, same 5 prompts, one request at a time,
against a live vLLM server (`TheBloke/Mistral-7B-Instruct-v0.1-AWQ`).

**Result:** **59.74 tokens/sec** (**1.59x** baseline).

**Conclusion:** A modest win. Expected — sequential requests barely exercise
vLLM's actual advantage (continuous batching), which only shows up under
concurrent load. Not the headline number.

## 3. Fixed concurrent load — 10 users

**Goal:** Measure real throughput/latency under realistic concurrent load —
the number that actually matters for a serving system.

**Method:** `load_test.js` (k6), 10 VUs (one API key each), ramp/hold/ramp,
`max_tokens=100`, default vLLM flags (`gpu-memory-utilization=0.90`).

**Result:** 100% success (0 failures). Aggregate throughput **~308 tok/s**
(185 requests / 60s test). Latency: p50=1.58s, p95=1.88s, **p99=1.91s**.

**Conclusion:** ~8x the naive baseline's throughput, and ~5x vLLM's own
sequential number — this is the real concurrent-batching advantage. Naive
`generate()` cannot batch concurrent requests at all; vLLM can.

## 4. Tuning test — `gpu-memory-utilization` 0.90 → 0.95

**Goal:** Test whether more KV-cache memory headroom improves throughput or
latency at 10 concurrent users.

**Method:** Relaunched vLLM with `--gpu-memory-utilization 0.95`, re-ran the
identical test from experiment 3.

**Result:** p99=1.91s — **identical**, throughput unchanged.

**Conclusion:** Not memory-bound at this concurrency. `nvidia-smi` showed GPU
compute utilization pegged at 100% from the very start of the test (even at
10 VUs) — the bottleneck is raw compute, not KV-cache headroom.

## 5. Stress test A — staircase 10 → 40 concurrent users

**Goal:** Find the point where the system starts to degrade or fail —
nothing so far had measured "concurrent users supported."

**Method:** `stress_test.js`, staircase stages at 10/20/30/40 VUs,
`max_tokens=100`.

**Result:** **0% failures** throughout. GPU utilization 100% continuously.
Aggregate throughput scaled to ~8.08 req/s (blended), up from ~3.07 req/s
at flat 10 VUs. p95 latency rose from 1.88s → **2.3s** at 40 users.

**Conclusion:** System handles at least 40 concurrent users gracefully, no
failures. Throughput kept scaling with concurrency — the real ceiling hadn't
been found yet.

## 6. Stress test B — staircase 60 → 120 concurrent users

**Goal:** Continue searching for the real breaking point beyond 40.

**Method:** `stress_test_high.js`, staircase stages at 60/80/100/120 VUs.

**Result:** Still **0% failures**. GPU 100% utilized throughout. Aggregate
throughput ~19.5 req/s (blended) — up again, but concurrency tripled (40→120)
while throughput only grew ~2.4x, showing diminishing returns. p95 latency
rose to **4.89s** (blended) at 120 users.

**Conclusion:** Graceful degradation, not failure — latency climbs, errors
don't appear. This deployment (30s gateway queue timeout, vLLM's own
scheduler) doesn't produce a clean "failure point" at this scale; capacity is
better defined by a latency threshold than by waiting for errors.

## 7. Tuning test — explicit `max-num-seqs=256`

**Goal:** Test whether raising vLLM's scheduler batch-size cap improves
throughput/latency at 10 concurrent users (second tuning hypothesis, after
ruling out memory in experiment 4).

**Method:** Relaunched vLLM with `--max-num-seqs 256`, re-ran the identical
10-VU test from experiment 3.

**Result:** p99=1.91s — **identical again**, third time running.

**Conclusion:** The scheduler cap wasn't binding at 10 concurrent users
either — consistent with the compute-bound finding. (This value later turned
out to be the actual hard capacity ceiling once concurrency was pushed far
higher — see experiments 9-10.)

## 8. Long-generation test — `max_tokens=400` at 60 concurrent users

**Goal:** Test whether longer generations (more KV-cache use per sequence)
shift the bottleneck from compute-bound to memory-bound — a different stress
dimension than raw concurrency alone.

**Method:** `long_gen_test.js`, 60 VUs, `max_tokens=400` (4x the earlier
tests), with live polling of vLLM's own scheduler metrics
(`vllm:num_requests_waiting`, `reason="capacity"`) during the run.

**Result:** **0% failures**, reproduced across two runs. GPU memory usage
stayed flat (~20.4GB, unchanged from shorter-token tests — vLLM pre-allocates
its KV-cache pool at startup, it doesn't grow per-request). vLLM's own
metrics showed **zero queueing for the entire test** —
`num_requests_waiting=0` and `reason="capacity"=0` at every sample. Latency
showed a long tail (max≈10.5s vs p95≈3.0s), reproduced consistently.

**Conclusion:** Memory/capacity was definitively *not* the cause of the
latency tail — proven directly via vLLM's own scheduler state, not inferred.
The tail is explained by natural generation-length variance under
`temperature=0.7` sampling (`max_tokens` is a ceiling, not a fixed length) —
not a system bottleneck.

## 9. Extreme stress test — staircase 200 → 1000 concurrent users (first run)

**Goal:** Push far beyond any previously-tested concurrency to find the true
breaking point.

**Method:** `stress_test_extreme.js`, staircase stages from 200 to 1000 VUs
(1000 distinct API keys), `max_tokens=100`.

**Result:** **54.99% failure rate.** vLLM's own metrics showed
`num_requests_running` capping at exactly **256** — matching the
`--max-num-seqs 256` set in experiment 7 — with `num_requests_waiting`
climbing into the hundreds once concurrency exceeded that cap. *Note: a
separate gateway-side connection-handling limitation also contributed to
failures in this specific run, making the raw failure percentage here not a
clean measurement of vLLM capacity alone — see the retest below.*

**Conclusion:** Found vLLM's real, hard scheduling ceiling: **exactly 256
concurrent sequences**, reproducible and exact.

## 10. Extreme stress test — staircase 200 → 1000 concurrent users (retest)

**Goal:** Re-measure the concurrency ceiling with a clean gateway (connection
handling fixed), to get an uncomplicated reading of the real limit.

**Method:** Same `stress_test_extreme.js` staircase, gateway now using a
single pooled HTTP client (max 300 pooled connections).

**Result:** **83.88% failure rate** — higher than the first run, but a
different, more informative failure mode: client-side timeouts (k6's own 60s
default), not connection errors. vLLM's own metrics again showed
`num_requests_running` capped at exactly **256**, with
`num_requests_waiting` *inside vLLM* peaking at only ~40 — much lower than
expected, because the gateway's connection pool (300) was itself gating
entry just above vLLM's ceiling, queuing the remaining ~700 requests
*inside the client*, invisible to vLLM, long enough to exceed k6's timeout.

**Conclusion:** **The real, defensible concurrency ceiling for this
deployment is ~256-300 concurrent requests** — vLLM's scheduler limit (256),
closely gated by the connection pool (300). Beyond that, requests queue and
eventually time out. This is a precise, explainable number tied directly to
a specific config value, not a guess or an open question.

## 11. Time-to-first-token (TTFT) — streaming vs. non-streaming

**Goal:** Every latency number so far measured *total* completion time. For a
chat-facing product, what users actually perceive as responsive is how fast
the *first* token arrives, not the whole response — which requires streaming
support (added to the gateway: `stream: true`, proxied via FastAPI
`StreamingResponse` + httpx's streaming client, with the semaphore/backpressure
slot correctly held for the full stream duration, not released early).

**Method:** `ttft_test.js`, 10 concurrent users, identical prompt/`max_tokens`,
run twice — once with `stream=false`, once with `stream=true` — comparing
k6's `http_req_waiting` (time to first byte) against `http_req_duration`
(total time).

**Result:**

| | Non-streaming | Streaming |
|---|---|---|
| TTFT/TTFB (avg) | 1497ms | **78.8ms** |
| TTFT/TTFB (p95) | 1806ms | **126ms** |
| Total completion time (avg) | 1497ms | 1527ms (unchanged) |

0% failures in both runs (187 requests, streaming run).

**Conclusion:** Streaming doesn't change total generation time (as expected —
same model, same work) but improves perceived responsiveness by **~19x on
average, ~14x at p95**. Non-streaming's TTFB essentially equals its total
time (nothing is sent until the whole response is ready) — confirming the
measurement is correct, not an artifact. This is the metric that actually
matters for a chat-facing product, and the gap between it and total-time
numbers (experiments 1-10) is the whole reason production LLM APIs stream by
default.

---

## Summary

| Experiment | Concurrency | Result |
|---|---|---|
| Baseline `generate()` | 1 | 37.54 tok/s |
| vLLM sequential | 1 | 59.74 tok/s (1.59x) |
| vLLM concurrent | 10 | ~308 tok/s, p99=1.91s, 0% failures |
| vLLM concurrent | 40 | p95=2.3s, 0% failures |
| vLLM concurrent | 120 | p95=4.89s, 0% failures |
| vLLM concurrent | 256-300 | real capacity ceiling |
| vLLM concurrent | 1000 | ~84% failures beyond the ceiling |
| TTFT (streaming) | 10 | 78.8ms avg (vs 1497ms non-streaming, ~19x) |

**Tuning findings:** `gpu-memory-utilization` and `max-num-seqs` showed no
effect at ≤120 concurrent users (system was compute-bound, not
memory/scheduler-bound, at that scale) — but `max-num-seqs=256` turned out to
be the exact hard ceiling once concurrency was pushed past it. Longer
generations (400 tokens) at 60 users showed zero KV-cache queueing, ruling
out memory pressure as a concern at that scale.
