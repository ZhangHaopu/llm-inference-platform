// Load test for the LLM inference gateway.
//
// Run with:
//   API_KEYS=key1,key2,key3 k6 run loadtest/load_test.js
// Push results into the project's own Prometheus (docker compose must be up
// with prometheus's remote-write receiver enabled):
//   API_KEYS=key1,key2,key3 k6 run -o experimental-prometheus-rw loadtest/load_test.js

import http from "k6/http";
import { check, sleep } from "k6";

const GATEWAY_URL = __ENV.GATEWAY_URL || "http://localhost:8080";
const API_KEYS = (__ENV.API_KEYS || "secret123").split(",");
const MODEL = __ENV.MODEL || "mistralai/Mistral-7B-Instruct-v0.1";

export const options = {
  stages: [
    { duration: "15s", target: 10 }, // ramp up to 10 concurrent VUs
    { duration: "30s", target: 10 }, // hold at 10 VUs
    { duration: "15s", target: 0 }, // ramp down
  ],
  thresholds: {
    http_req_duration: ["p(99)<500"], // CV target: p99 under 500ms
    checks: ["rate>0.95"], // fewer than 5% failed requests
  },
};

export default function () {
  // Each VU gets its own API key so the rate limiter (per-key) doesn't
  // dominate the results — this measures total system capacity across many
  // simulated users, not one client's rate limit.
  const apiKey = API_KEYS[(__VU - 1) % API_KEYS.length];

  const payload = JSON.stringify({
    model: MODEL,
    prompt: "What is machine learning?",
    max_tokens: 100,
    temperature: 0.7,
  });

  const params = {
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": apiKey,
    },
  };

  const res = http.post(`${GATEWAY_URL}/v1/completions`, payload, params);

  check(res, {
    "status is 200": (r) => r.status === 200,
  });

  // Simulate think-time between a user's requests. Without this, each VU
  // hammers the gateway back-to-back with no pacing, which blows straight
  // through the per-key rate limiter and mostly just measures 429s instead
  // of real system capacity.
  sleep(1);
}
