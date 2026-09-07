"""Per-API-key rate limiting using a token bucket.

In-memory state only works correctly for a single gateway process. Running
multiple uvicorn workers would give each worker its own bucket, effectively
multiplying the limit — a real deployment would back this with Redis instead.
"""

import time

from fastapi import Depends, HTTPException

from auth import get_api_key

CAPACITY = 5
REFILL_RATE = 1.0  # tokens per second

buckets: dict[str, tuple[float, float]] = {}  # api_key -> (tokens, last_refill)


def _consume(api_key: str) -> bool:
    now = time.monotonic()
    tokens, last_refill = buckets.get(api_key, (CAPACITY, now))

    elapsed = now - last_refill
    tokens = min(CAPACITY, tokens + elapsed * REFILL_RATE)

    if tokens >= 1:
        buckets[api_key] = (tokens - 1, now)
        return True

    buckets[api_key] = (tokens, now)
    return False


def check_rate_limit(api_key: str = Depends(get_api_key)) -> None:
    if not _consume(api_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
