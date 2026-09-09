"""FastAPI gateway in front of the vLLM OpenAI-compatible server."""

import os

import httpx
from fastapi import Depends, FastAPI, HTTPException, Response
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from rate_limit import check_rate_limit

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000")

app = FastAPI(title="LLM Inference Gateway")

REQUEST_COUNT = Counter(
    "gateway_requests_total", "Total completion requests", ["status"]
)
REQUEST_LATENCY = Histogram(
    "gateway_request_latency_seconds", "Latency of completion requests"
)


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int
    temperature: float = 0.7


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/completions")
async def create_completion(
    request: CompletionRequest, _: None = Depends(check_rate_limit)
) -> dict:
    with REQUEST_LATENCY.time():
        async with httpx.AsyncClient(timeout=300) as client:
            try:
                response = await client.post(
                    f"{VLLM_URL}/v1/completions",
                    json=request.model_dump(),
                )
            except httpx.RequestError as exc:
                REQUEST_COUNT.labels(status="error").inc()
                raise HTTPException(
                    status_code=502, detail=f"Could not reach vLLM server: {exc}"
                ) from exc

        if response.status_code != 200:
            REQUEST_COUNT.labels(status="error").inc()
            raise HTTPException(status_code=response.status_code, detail=response.text)

    REQUEST_COUNT.labels(status="success").inc()
    return response.json()
