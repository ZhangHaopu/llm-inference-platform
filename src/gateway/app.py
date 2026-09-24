"""FastAPI gateway in front of the vLLM OpenAI-compatible server."""

import asyncio
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
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

http_client: httpx.AsyncClient = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    # A single shared, pooled client instead of one per request — creating a
    # new AsyncClient (and its own connection pool) per request exhausts the
    # container's file descriptor limit under high concurrency (hit this at
    # ~1000 concurrent requests: "OSError: [Errno 24] Too many open files").
    http_client = httpx.AsyncClient(
        timeout=300,
        limits=httpx.Limits(max_connections=300, max_keepalive_connections=50),
    )
    yield
    await http_client.aclose()


app = FastAPI(title="LLM Inference Gateway", lifespan=lifespan)

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
    stream: bool = False


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def _stream_upstream(response: httpx.Response, stream_ctx, start_time: float):
    # Runs *after* create_completion returns the StreamingResponse — the
    # semaphore slot and IN_FLIGHT count must stay held until the stream
    # actually finishes, not until the endpoint function returns, otherwise
    # backpressure would stop working the moment streaming starts.
    try:
        async for chunk in response.aiter_bytes():
            yield chunk
        REQUEST_COUNT.labels(status="success").inc()
    except httpx.RequestError:
        REQUEST_COUNT.labels(status="error").inc()
    finally:
        await stream_ctx.__aexit__(None, None, None)
        REQUEST_LATENCY.observe(time.perf_counter() - start_time)
        semaphore.release()
        IN_FLIGHT.dec()


@app.post("/v1/completions")
async def create_completion(
    request: CompletionRequest, _: None = Depends(check_rate_limit)
):
    IN_FLIGHT.inc()

    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=QUEUE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        IN_FLIGHT.dec()
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(
            status_code=503, detail="Server busy, try again later"
        ) from None

    if not request.stream:
        with REQUEST_LATENCY.time():
            try:
                try:
                    response = await http_client.post(
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
                IN_FLIGHT.dec()

        REQUEST_COUNT.labels(status="success").inc()
        return response.json()

    # Streaming path: open the upstream connection and check its status
    # *before* committing to a StreamingResponse, but keep the connection
    # open across this function's return — closing it here would cut the
    # stream off before any chunks reach our own client.
    start_time = time.perf_counter()
    stream_ctx = http_client.stream(
        "POST", f"{VLLM_URL}/v1/completions", json=request.model_dump()
    )
    try:
        response = await stream_ctx.__aenter__()
    except httpx.RequestError as exc:
        semaphore.release()
        IN_FLIGHT.dec()
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(
            status_code=502, detail=f"Could not reach vLLM server: {exc}"
        ) from exc

    if response.status_code != 200:
        body = await response.aread()
        await stream_ctx.__aexit__(None, None, None)
        semaphore.release()
        IN_FLIGHT.dec()
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=response.status_code, detail=body.decode())

    return StreamingResponse(
        _stream_upstream(response, stream_ctx, start_time),
        media_type="text/event-stream",
    )
