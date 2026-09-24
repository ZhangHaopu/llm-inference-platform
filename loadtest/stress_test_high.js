// Continuation of stress_test.js — 10-40 VUs already proved 0% failures with
// the GPU pegged at 100% util the whole time. This picks up from 60 VUs to
// find where it actually breaks, instead of re-running the already-proven
// lower stages.
//
// Run with:
//   API_KEYS=k1,...,k120 k6 run -o experimental-prometheus-rw loadtest/stress_test_high.js

import http from "k6/http";
import { check, sleep } from "k6";

const GATEWAY_URL = __ENV.GATEWAY_URL || "http://localhost:8080";
const API_KEYS = (__ENV.API_KEYS || "secret123").split(",");
const MODEL = __ENV.MODEL || "mistralai/Mistral-7B-Instruct-v0.1";

export const options = {
  stages: [
    { duration: "20s", target: 60 },
    { duration: "30s", target: 60 },
    { duration: "20s", target: 80 },
    { duration: "30s", target: 80 },
    { duration: "20s", target: 100 },
    { duration: "30s", target: 100 },
    { duration: "20s", target: 120 },
    { duration: "30s", target: 120 },
    { duration: "20s", target: 0 },
  ],
  thresholds: {
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
