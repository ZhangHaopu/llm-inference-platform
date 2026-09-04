"""Benchmark a running vLLM OpenAI-compatible completion endpoint."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests

TEST_PROMPTS = [
    "What is machine learning?",
    "Explain how neural networks work in simple terms.",
    "Write a brief Python function that calculates factorial.",
    "Describe the process of photosynthesis in plants.",
    "What are the main differences between supervised and unsupervised learning?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000/v1/completions")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--runs", type=int, default=len(TEST_PROMPTS))
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--output", default="benchmarks/vllm.json")
    return parser.parse_args()


def run_benchmark(
    url: str,
    model: str,
    num_runs: int,
    max_tokens: int,
) -> dict[str, Any]:
    """Send prompts sequentially and measure end-to-end request performance."""
    prompts = TEST_PROMPTS[: min(num_runs, len(TEST_PROMPTS))]
    results: list[dict[str, Any]] = []
    total_tokens = 0
    total_time = 0.0

    for index, prompt in enumerate(prompts, start=1):
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }

        start_time = time.perf_counter()
        response = requests.post(url, json=payload, timeout=300)
        elapsed_seconds = time.perf_counter() - start_time
        response.raise_for_status()
        body = response.json()

        usage = body.get("usage", {})
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is None:
            completion_tokens = len(body["choices"][0]["text"].split())

        requests_per_second = 1 / elapsed_seconds if elapsed_seconds else 0
        tokens_per_second = completion_tokens / elapsed_seconds if elapsed_seconds else 0
        results.append(
            {
                "run": index,
                "prompt": prompt,
                "completion_tokens": completion_tokens,
                "elapsed_seconds": round(elapsed_seconds, 3),
                "tokens_per_second": round(tokens_per_second, 2),
            }
        )
        total_tokens += completion_tokens
        total_time += elapsed_seconds

    return {
        "endpoint": url,
        "model": model,
        "num_runs": len(prompts),
        "max_tokens_per_prompt": max_tokens,
        "runs": results,
        "summary": {
            "total_completion_tokens": total_tokens,
            "total_time_seconds": round(total_time, 3),
            "aggregate_tokens_per_second": round(
                total_tokens / total_time if total_time else 0, 2
            ),
            "average_request_latency_seconds": round(
                total_time / len(prompts) if prompts else 0, 3
            ),
        },
    }


def save_results(results: dict[str, Any], output_file: str) -> Path:
    """Write benchmark results as formatted JSON."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    return output_path


def main() -> None:
    args = parse_args()
    results = run_benchmark(
        url=args.url,
        model=args.model,
        num_runs=args.runs,
        max_tokens=args.max_tokens,
    )
    output_path = save_results(results, args.output)
    summary = results["summary"]
    print(f"Aggregate throughput: {summary['aggregate_tokens_per_second']} tokens/sec")
    print(f"Average latency: {summary['average_request_latency_seconds']} sec")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
