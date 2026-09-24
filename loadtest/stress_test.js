// Stress / breaking-point test for the LLM inference gateway.
//
// Unlike load_test.js (fixed concurrency, steady state), this steps VU count
// up in stages until latency or error rate degrades — answering "how many
// concurrent users can this actually support?" rather than "what's normal
// latency at a known-good concurrency?"
//
// Run with:
//   API_KEYS=k1,...,k40 k6 run -o experimental-prometheus-rw loadtest/stress_test.js

import http from "k6/http";
import { check, sleep } from "k6";

const GATEWAY_URL = __ENV.GATEWAY_URL || "http://localhost:8080";
const API_KEYS = (__ENV.API_KEYS || "secret123").split(",");
const MODEL = __ENV.MODEL || "mistralai/Mistral-7B-Instruct-v0.1";

export const options = {
  stages: [
    { duration: "20s", target: 10 },
    { duration: "30s", target: 10 },
    { duration: "20s", target: 20 },
    { duration: "30s", target: 20 },
    { duration: "20s", target: 30 },
    { duration: "30s", target: 30 },
    { duration: "20s", target: 40 },
    { duration: "30s", target: 40 },
    { duration: "20s", target: 0 },
  ],
  thresholds: {
    // Don't abort the run on threshold breach — we WANT to see degradation,
    // that's the point of a stress test. abortOnFail stays false (default).
    http_req_failed: ["rate<1"],
  },
};

export default function () {
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

  sleep(1);
}
