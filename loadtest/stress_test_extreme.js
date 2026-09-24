// Continuation of stress_test.js / stress_test_high.js. Already proven:
// 10-120 VUs = 0% failures, GPU compute-bound the whole time, vLLM never
// queued a single request. This pushes further to find where (if anywhere)
// a real ceiling exists.
import http from "k6/http";
import { check, sleep } from "k6";

const GATEWAY_URL = __ENV.GATEWAY_URL || "http://localhost:8080";
const API_KEYS = (__ENV.API_KEYS || "secret123").split(",");
const MODEL = __ENV.MODEL || "mistralai/Mistral-7B-Instruct-v0.1";

export const options = {
  stages: [
    { duration: "20s", target: 200 },
    { duration: "25s", target: 200 },
    { duration: "20s", target: 400 },
    { duration: "25s", target: 400 },
    { duration: "20s", target: 600 },
    { duration: "25s", target: 600 },
    { duration: "20s", target: 800 },
    { duration: "25s", target: 800 },
    { duration: "20s", target: 1000 },
    { duration: "25s", target: 1000 },
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
