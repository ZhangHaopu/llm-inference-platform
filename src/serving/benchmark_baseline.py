"""
Baseline benchmark for naive model.generate() inference.

This measures tokens/sec with basic Hugging Face inference (no vLLM optimization).
Used as a "before" measurement for Step 2 (vLLM integration).

Results saved to benchmarks/baseline.json
"""

import time
import json
from pathlib import Path
from model_loader import load_model_and_tokenizer, generate_text

# Test prompts of varying lengths
TEST_PROMPTS = [
    "What is machine learning?",
    "Explain how neural networks work in simple terms.",
    "Write a brief Python function that calculates factorial.",
    "Describe the process of photosynthesis in plants.",
    "What are the main differences between supervised and unsupervised learning?",
]

def benchmark_generation(num_runs: int = 5, max_tokens: int = 100) -> dict:
    """
    Benchmark model.generate() performance.
    
    Args:
        num_runs: Number of test prompts to run
        max_tokens: Max tokens per generation
    
    Returns:
        Dictionary with timing and throughput metrics
    """
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    
    results = {
        "num_prompts": min(num_runs, len(TEST_PROMPTS)),
        "max_tokens_per_prompt": max_tokens,
        "runs": [],
        "summary": {}
    }
    
    print(f"Running benchmark ({min(num_runs, len(TEST_PROMPTS))} prompts, max {max_tokens} tokens each)...\n")
    
    total_tokens = 0
    total_time = 0.0
    
    for i, prompt in enumerate(TEST_PROMPTS[:num_runs]):
        print(f"[{i+1}/{min(num_runs, len(TEST_PROMPTS))}] Prompt: {prompt[:50]}...")
        
        # Time the generation
        start_time = time.time()
        result = generate_text(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            temperature=0.7
        )
        end_time = time.time()
        
        # Count generated tokens (rough: tokenize the output)
        generated_text = result.split(prompt)[-1]  # Get only new tokens
        tokens_generated = len(tokenizer.encode(generated_text))
        
        elapsed_time = end_time - start_time
        tokens_per_sec = tokens_generated / elapsed_time if elapsed_time > 0 else 0
        
        run_result = {
            "prompt": prompt,
            "tokens_generated": tokens_generated,
            "elapsed_seconds": round(elapsed_time, 2),
            "tokens_per_sec": round(tokens_per_sec, 2),
        }
        results["runs"].append(run_result)
        
        total_tokens += tokens_generated
        total_time += elapsed_time
        
        print(f"  ✓ {tokens_generated} tokens in {elapsed_time:.2f}s ({tokens_per_sec:.2f} tokens/sec)\n")
    
    # Calculate summary statistics
    avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    results["summary"] = {
        "total_tokens": total_tokens,
        "total_time_seconds": round(total_time, 2),
        "average_tokens_per_sec": round(avg_tokens_per_sec, 2),
    }
    
    return results


def save_results(results: dict, output_dir: str = "benchmarks") -> Path:
    """
    Save benchmark results to JSON.
    
    Args:
        results: Benchmark results dictionary
        output_dir: Directory to save results
    
    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results_file = output_path / "baseline.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    return results_file


if __name__ == "__main__":
    # Run benchmark
    results = benchmark_generation(num_runs=5, max_tokens=100)
    
    # Display summary
    print("\n" + "="*60)
    print("BASELINE BENCHMARK SUMMARY")
    print("="*60)
    print(f"Total tokens generated: {results['summary']['total_tokens']}")
    print(f"Total time: {results['summary']['total_time_seconds']:.2f}s")
    print(f"Average throughput: {results['summary']['average_tokens_per_sec']:.2f} tokens/sec")
    print("="*60 + "\n")
    
    # Save results
    results_file = save_results(results)
    print(f"✓ Results saved to {results_file}")
