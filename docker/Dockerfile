# Gateway image: lightweight FastAPI proxy, no torch/vllm/transformers.
# The vLLM server itself runs from the official vllm/vllm-openai image
# instead of being built here.
FROM python:3.9-slim

WORKDIR /app

COPY requirements-gateway.txt .
RUN pip install --no-cache-dir -r requirements-gateway.txt

COPY src/gateway/ .

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
