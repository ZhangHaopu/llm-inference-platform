"""Launch the vLLM OpenAI-compatible inference server."""

import argparse
import subprocess
import sys

import torch

DEFAULT_MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.1-AWQ"
DEFAULT_QUANTIZATION = "awq"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--quantization", default=DEFAULT_QUANTIZATION)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        print(
            "vLLM requires a CUDA-capable GPU for this deployment. "
            "Run this launcher on a Linux GPU instance, such as RunPod or Colab.",
            file=sys.stderr,
        )
        return 1

    command = [
        "vllm",
        "serve",
        args.model,
        "--quantization",
        args.quantization,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.max_num_seqs is not None:
        command += ["--max-num-seqs", str(args.max_num_seqs)]

    print("Starting vLLM:", " ".join(command))
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
