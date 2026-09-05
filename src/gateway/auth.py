"""API-key authentication for the gateway.

Keys are read from the API_KEYS env var (comma-separated). Good enough for
a v1 single-tenant gateway; a real multi-tenant service would store hashed
keys in a database instead of a plaintext env var.
"""

import os
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

VALID_API_KEYS = {
    key.strip() for key in os.environ.get("API_KEYS", "").split(",") if key.strip()
}


def get_api_key(api_key: Optional[str] = Depends(api_key_header)) -> str:
    if api_key is None or api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")
    return api_key
