// Tests whether longer generations (more KV cache pressure per sequence)
// shift the bottleneck from compute-bound (confirmed at max_tokens=100) to
// memory-bound, at a concurrency already proven safe at the shorter length.
import http from "k6/http";
import { check, sleep } from "k6";

const GATEWAY_URL = __ENV.GATEWAY_URL || "http://localhost:8080";
const API_KEYS = (__ENV.API_KEYS || "secret123").split(",");
const MODEL = __ENV.MODEL || "mistralai/Mistral-7B-Instruct-v0.1";
const MAX_TOKENS = parseInt(__ENV.MAX_TOKENS || "400", 10);

export const options = {
  stages: [
    { duration: "20s", target: 60 },
    { duration: "90s", target: 60 },
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
    max_tokens: MAX_TOKENS,
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
