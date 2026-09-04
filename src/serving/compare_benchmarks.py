"""Compare baseline (naive generate()) vs vLLM benchmark results.

Reads the JSON outputs of benchmark_baseline.py and benchmark_vllm.py and
reports the throughput delta between them, matching the "Benchmark Results"
table in the README.
"""

import argparse
import json
from pathlib import Path
from typing import Any

# Each entry: JSON key to look for, in priority order, in a results file's
# "summary" block. The two benchmark scripts don't share a schema (baseline
# predates the vLLM client and measures local generate() calls rather than
# HTTP requests), so throughput is read by trying each known field name.
THROUGHPUT_KEYS = ("aggregate_tokens_per_second", "average_tokens_per_sec")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="benchmarks/baseline.json")
    parser.add_argument("--vllm", default="benchmarks/vllm.json")
    parser.add_argument("--output", default="benchmarks/results.md")
    return parser.parse_args()


def load_results(path: str, label: str) -> dict[str, Any]:
    results_path = Path(path)
    if not results_path.exists():
        raise FileNotFoundError(
            f"No {label} results at {results_path}. Run the {label} benchmark "
            "on a GPU host first."
        )
    return json.loads(results_path.read_text())


def extract_throughput(results: dict[str, Any], label: str) -> float:
    summary = results.get("summary", {})
    for key in THROUGHPUT_KEYS:
        if key in summary:
            return summary[key]
    raise KeyError(f"Could not find a tokens/sec field in {label} summary: {summary}")


def compare(baseline: dict[str, Any], vllm: dict[str, Any]) -> dict[str, Any]:
    baseline_tps = extract_throughput(baseline, "baseline")
    vllm_tps = extract_throughput(vllm, "vLLM")
    speedup = vllm_tps / baseline_tps if baseline_tps else 0
    return {
        "baseline_tokens_per_second": baseline_tps,
        "vllm_tokens_per_second": vllm_tps,
        "speedup": round(speedup, 2),
    }


def render_markdown(comparison: dict[str, Any]) -> str:
    return (
        "## Baseline vs. vLLM Throughput\n\n"
        "| Metric | Baseline (`generate()`) | vLLM | Speedup |\n"
        "|--------|--------------------------|------|---------|\n"
        f"| Tokens/sec | {comparison['baseline_tokens_per_second']} | "
        f"{comparison['vllm_tokens_per_second']} | "
        f"{comparison['speedup']}x |\n"
    )


def main() -> None:
    args = parse_args()
    baseline = load_results(args.baseline, "baseline")
    vllm = load_results(args.vllm, "vLLM")

    comparison = compare(baseline, vllm)
    markdown = render_markdown(comparison)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown)

    print(markdown)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
