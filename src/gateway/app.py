"""FastAPI gateway in front of the vLLM OpenAI-compatible server."""

import asyncio
import os

import httpx
from fastapi import Depends, FastAPI, HTTPException, Response
from pydantic import BaseModel
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from rate_limit import check_rate_limit

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000")
MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "10"))
QUEUE_TIMEOUT_SECONDS = float(os.environ.get("QUEUE_TIMEOUT_SECONDS", "30"))

app = FastAPI(title="LLM Inference Gateway")

semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

REQUEST_COUNT = Counter(
    "gateway_requests_total", "Total completion requests", ["status"]
)
REQUEST_LATENCY = Histogram(
    "gateway_request_latency_seconds", "Latency of completion requests"
)
IN_FLIGHT = Gauge(
    "gateway_in_flight_requests",
    "Requests currently queued or being forwarded to vLLM",
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
    IN_FLIGHT.inc()
    try:
        with REQUEST_LATENCY.time():
            try:
                await asyncio.wait_for(
                    semaphore.acquire(), timeout=QUEUE_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                REQUEST_COUNT.labels(status="error").inc()
                raise HTTPException(
                    status_code=503, detail="Server busy, try again later"
                ) from None

            try:
                async with httpx.AsyncClient(timeout=300) as client:
                    try:
                        response = await client.post(
                            f"{VLLM_URL}/v1/completions",
                            json=request.model_dump(),
                        )
                    except httpx.RequestError as exc:
                        REQUEST_COUNT.labels(status="error").inc()
                        raise HTTPException(
                            status_code=502,
                            detail=f"Could not reach vLLM server: {exc}",
                        ) from exc

                if response.status_code != 200:
                    REQUEST_COUNT.labels(status="error").inc()
                    raise HTTPException(
                        status_code=response.status_code, detail=response.text
                    )
            finally:
                semaphore.release()
    finally:
        IN_FLIGHT.dec()

    REQUEST_COUNT.labels(status="success").inc()
    return response.json()
