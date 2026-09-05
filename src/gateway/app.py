"""FastAPI gateway in front of the vLLM OpenAI-compatible server."""

import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000")

app = FastAPI(title="LLM Inference Gateway")


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int
    temperature: float = 0.7


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest) -> dict:
    async with httpx.AsyncClient(timeout=300) as client:
        try:
            response = await client.post(
                f"{VLLM_URL}/v1/completions",
                json=request.model_dump(),
            )
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=502, detail=f"Could not reach vLLM server: {exc}"
            ) from exc

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    return response.json()
