// Compares perceived responsiveness (TTFB, k6's http_req_waiting) against
// total completion time (http_req_duration) for streaming vs non-streaming
// requests at the same concurrency.
import http from "k6/http";
import { check, sleep } from "k6";

const GATEWAY_URL = __ENV.GATEWAY_URL || "http://localhost:8080";
const API_KEYS = (__ENV.API_KEYS || "secret123").split(",");
const MODEL = __ENV.MODEL || "mistralai/Mistral-7B-Instruct-v0.1";
const STREAM = (__ENV.STREAM || "false") === "true";

export const options = {
  stages: [
    { duration: "15s", target: 10 },
    { duration: "30s", target: 10 },
    { duration: "15s", target: 0 },
  ],
};

export default function () {
  const apiKey = API_KEYS[(__VU - 1) % API_KEYS.length];

  const payload = JSON.stringify({
    model: MODEL,
    prompt: "What is machine learning?",
    max_tokens: 100,
    temperature: 0.7,
    stream: STREAM,
  });

  const params = {
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": apiKey,
    },
  };

  const res = http.post(`${GATEWAY_URL}/v1/completions`, payload, params);

  check(res, { "status is 200": (r) => r.status === 200 });

  sleep(1);
}
