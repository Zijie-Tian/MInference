# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Benchmark for MInference + vLLM V1 with LMCache KV cache offload.

This benchmark measures prefill latency with:
- MInference sparse attention
- LMCache CPU offload for prefix caching
- Tensor parallelism
- Cold vs Warm cache comparison (to show LMCache prefix cache benefit)

LMCache Key Concepts:
- Chunk-based caching: KV cache is stored in 256-token chunks
- Multi-level storage: GPU -> CPU -> Disk -> Remote
- Async offload: Does not block inference
- Cross-request reuse: Same prefix can be reused across requests

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python experiments/benchmarks/benchmark_e2e_vllm_offload.py \
        --run_benchmark --model_name /path/to/model
"""

import argparse
import os
import time
from pathlib import Path

# =============================================================================
# LMCache Configuration (MUST be set before importing vllm)
# =============================================================================
# LMCACHE_USE_EXPERIMENTAL: Enable LMCache V1 (required for vLLM V1)
# LMCACHE_CHUNK_SIZE: Token chunk size for caching (default 256)
# LMCACHE_LOCAL_CPU: Enable CPU memory backend
# LMCACHE_MAX_LOCAL_CPU_SIZE: CPU cache size in GB

os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
os.environ["LMCACHE_CHUNK_SIZE"] = "256"
os.environ["LMCACHE_LOCAL_CPU"] = "True"

# IMPORTANT: Set PYTHONHASHSEED for consistent hash across processes
# Without this, LMCache cache hits won't work correctly
os.environ["PYTHONHASHSEED"] = "0"

# vLLM V1 environment variables
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

import pandas as pd
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from minference import MInference

# Check LMCache availability
try:
    from vllm.config import KVTransferConfig

    import lmcache

    HAS_LMCACHE = True
    print(f"LMCache version: {lmcache.__version__ if hasattr(lmcache, '__version__') else 'unknown'}")
except ImportError:
    HAS_LMCACHE = False
    print("Warning: LMCache not available, running without CPU offload")

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "benchmark"


def run_target_length(
    m: int,
    llm,
    tokenizer,
    sampling_params,
    attn_type: str,
    num_iterations: int = 10,
    test_cache_hit: bool = False,
):
    """Run benchmark for a specific context length.

    Args:
        m: Target context length in tokens
        llm: vLLM LLM instance
        tokenizer: Tokenizer
        sampling_params: Sampling parameters
        attn_type: Attention type name for logging
        num_iterations: Number of iterations (excluding warmup)
        test_cache_hit: If True, run additional warm cache test

    Returns:
        If test_cache_hit is False: cold_latency (float)
        If test_cache_hit is True: (cold_latency, warm_latency) tuple
    """
    # Load prompt data
    prompt_file = Path(__file__).parent / "prompt_hardest.txt"
    if not prompt_file.exists():
        prompt_file = Path("./prompt_hardest.txt")
    prompt_complex = open(prompt_file).read()

    input_ids = tokenizer(prompt_complex)["input_ids"]
    n = len(input_ids)
    b = m // n + 1

    new_input_ids = (input_ids * b)[:m]
    prompt = tokenizer.decode(new_input_ids)

    # =========================================================================
    # Cold Cache Test: First run populates the cache
    # =========================================================================
    cold_times = []
    T = num_iterations
    for i in range(T + 1):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            outputs = llm.generate([prompt], sampling_params)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        if i == 0:
            # First run is both warmup AND cache population
            cold_first = elapsed
            print(f"  Cold (first): {cold_first:.4f}s")
        else:
            cold_times.append(elapsed)

    cold_avg = sum(cold_times) / len(cold_times)
    print(f"{attn_type} | {m} tokens | Cold avg: {cold_avg:.4f}s")

    if not test_cache_hit:
        return cold_avg

    # =========================================================================
    # Warm Cache Test: Same prompt to measure cache hit benefit
    # With LMCache, subsequent requests with same prefix should be faster
    # =========================================================================
    warm_times = []
    for i in range(T):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            # Use exact same prompt to test cache hit
            outputs = llm.generate([prompt], sampling_params)
        torch.cuda.synchronize()
        warm_times.append(time.time() - start)

    warm_avg = sum(warm_times) / len(warm_times)
    speedup = cold_avg / warm_avg if warm_avg > 0 else 0
    print(f"{attn_type} | {m} tokens | Warm avg: {warm_avg:.4f}s | Cache speedup: {speedup:.2f}x")

    return cold_avg, warm_avg


def create_llm(
    model_name: str,
    max_len: int,
    tensor_parallel_size: int,
    use_lmcache: bool,
    cpu_memory_gb: float,
    enable_prefix_caching: bool = True,
):
    """Create LLM instance with optional LMCache offload.

    Args:
        model_name: HuggingFace model name or local path
        max_len: Maximum model length
        tensor_parallel_size: Number of GPUs for tensor parallelism
        use_lmcache: Whether to enable LMCache
        cpu_memory_gb: CPU memory limit for LMCache in GB
        enable_prefix_caching: Whether to enable vLLM's built-in prefix caching
    """
    llm_kwargs = {
        "model": model_name,
        "enforce_eager": True,
        "max_model_len": max_len,
        "enable_chunked_prefill": False,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": 0.85,
        "enable_prefix_caching": enable_prefix_caching,
    }

    # Add KV cache offload if LMCache is available
    if use_lmcache and HAS_LMCACHE:
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(cpu_memory_gb)
        try:
            kv_config = KVTransferConfig(
                kv_connector="LMCacheConnectorV1",
                kv_role="kv_both",  # Both produce and consume KV cache
            )
            llm_kwargs["kv_transfer_config"] = kv_config
            print(f"LMCache enabled:")
            print(f"  - CPU memory: {cpu_memory_gb}GB")
            print(f"  - Chunk size: {os.environ.get('LMCACHE_CHUNK_SIZE', '256')} tokens")
            print(f"  - Connector: LMCacheConnectorV1")
        except Exception as e:
            print(f"Warning: Could not configure LMCache: {e}")
            print("Falling back to vLLM without LMCache")
    elif use_lmcache and not HAS_LMCACHE:
        print("Warning: LMCache requested but not installed")
        print("Install with: pip install lmcache")

    return LLM(**llm_kwargs)


def run_benchmark(
    model_name: str,
    tensor_parallel_size: int,
    use_lmcache: bool,
    cpu_memory_gb: float,
    target_lens: list = None,
    output_file: str = None,
    max_model_len: int = None,
    test_cache_hit: bool = True,
):
    """Run full benchmark across different context lengths.

    Args:
        model_name: Model path or HuggingFace model name
        tensor_parallel_size: Number of GPUs
        use_lmcache: Enable LMCache offload
        cpu_memory_gb: CPU memory for LMCache
        target_lens: List of context lengths to test
        output_file: Output CSV path
        max_model_len: Maximum model length (auto if None)
        test_cache_hit: Whether to test warm cache performance
    """
    if target_lens is None:
        target_lens = [l * 1024 for l in [4, 8, 16, 32, 64]]

    # For quick testing, can be overridden by attn_types argument
    ATTN_TYPES = ["flash_attn", "minference"]
    ATTN_TYPES2NAME = {
        "flash_attn": "FlashAttention-2",
        "minference": "MInference",
    }

    # Allow filtering attention types via environment variable for debugging
    env_attn_types = os.environ.get("ATTN_TYPES", None)
    if env_attn_types:
        ATTN_TYPES = [t.strip() for t in env_attn_types.split(",")]
        print(f"Using attention types from env: {ATTN_TYPES}")

    # Initialize results DataFrame
    results = []

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1)

    for attn_type in ATTN_TYPES:
        # Determine max_model_len: use provided value or cap at GPU memory limit
        if max_model_len is not None:
            max_len = max_model_len
        else:
            # Default: cap at 65536 (safe for 2x 24GB GPUs with TP=2)
            max_len = min(target_lens[-1] + 10_000, 65536)
        print(f"\n{'='*60}")
        print(f"Testing {ATTN_TYPES2NAME[attn_type]} (max_len={max_len})")
        print(f"LMCache: {'enabled' if use_lmcache and HAS_LMCACHE else 'disabled'}")
        print(f"{'='*60}")

        llm = create_llm(
            model_name, max_len, tensor_parallel_size, use_lmcache, cpu_memory_gb
        )

        if attn_type == "minference":
            minference_patch = MInference("vllm_minference", model_name)
            llm = minference_patch(llm)

        for context_len in target_lens:
            if context_len > max_len - 1000:
                print(f"Skipping {context_len} tokens (exceeds max_len)")
                results.append({
                    "context_length": context_len,
                    "context_length_k": f"{context_len // 1024}K",
                    "attn_type": ATTN_TYPES2NAME[attn_type],
                    "cold_latency_s": None,
                    "warm_latency_s": None,
                    "cache_speedup": None,
                    "throughput_tokens_per_s": None,
                })
                continue

            if test_cache_hit and use_lmcache and HAS_LMCACHE:
                cold_latency, warm_latency = run_target_length(
                    context_len, llm, tokenizer, sampling_params, attn_type,
                    test_cache_hit=True
                )
                cache_speedup = cold_latency / warm_latency if warm_latency > 0 else 0
            else:
                cold_latency = run_target_length(
                    context_len, llm, tokenizer, sampling_params, attn_type,
                    test_cache_hit=False
                )
                warm_latency = None
                cache_speedup = None

            throughput = context_len / cold_latency if cold_latency > 0 else 0

            results.append({
                "context_length": context_len,
                "context_length_k": f"{context_len // 1024}K",
                "attn_type": ATTN_TYPES2NAME[attn_type],
                "cold_latency_s": cold_latency,
                "warm_latency_s": warm_latency,
                "cache_speedup": cache_speedup,
                "throughput_tokens_per_s": throughput,
            })
            torch.cuda.empty_cache()

        del llm
        torch.cuda.empty_cache()

    # Create DataFrame
    df = pd.DataFrame(results)

    # Print results in a nice format
    print("\n" + "=" * 70)
    print("RESULTS (Prefill Latency)")
    print("=" * 70)

    # Pivot table for display
    pivot_cold = df.pivot(
        index="context_length_k", columns="attn_type", values="cold_latency_s"
    )
    print("\nCold Latency (seconds) - First request, no cache:")
    print(pivot_cold.to_string())

    if test_cache_hit and use_lmcache and HAS_LMCACHE:
        pivot_warm = df.pivot(
            index="context_length_k", columns="attn_type", values="warm_latency_s"
        )
        print("\nWarm Latency (seconds) - With LMCache prefix cache hit:")
        print(pivot_warm.to_string())

        pivot_speedup = df.pivot(
            index="context_length_k", columns="attn_type", values="cache_speedup"
        )
        print("\nCache Speedup (cold / warm):")
        print(pivot_speedup.to_string())

    pivot_throughput = df.pivot(
        index="context_length_k", columns="attn_type", values="throughput_tokens_per_s"
    )
    print("\nThroughput (tokens/s, based on cold latency):")
    print(pivot_throughput.to_string())

    # Calculate MInference speedup over FlashAttention
    if "FlashAttention-2" in pivot_cold.columns and "MInference" in pivot_cold.columns:
        speedup = pivot_cold["FlashAttention-2"] / pivot_cold["MInference"]
        print("\nMInference Speedup over FlashAttention-2 (cold):")
        print(speedup.to_string())

    # Save results
    if output_file is None:
        output_file = RESULTS_DIR / "vllm_lmcache_perf.csv"
    else:
        output_file = Path(output_file)

    # Ensure directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # Also save pivot tables
    pivot_file = output_file.parent / "vllm_lmcache_perf_pivot.csv"
    pivot_cold.to_csv(pivot_file)
    print(f"Pivot table saved to {pivot_file}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark MInference + vLLM with LMCache offload"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="/home/zijie/models/Llama-3-8B-Instruct-262k",
        help="Model path or HuggingFace model name",
    )
    parser.add_argument(
        "--attn_type",
        type=str,
        choices=["flash_attn", "minference"],
        help="Attention type for single run",
    )
    parser.add_argument(
        "--context_window",
        type=int,
        default=100_000,
        help="Context window for single run",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=2,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--no_lmcache",
        action="store_true",
        help="Disable LMCache CPU offload",
    )
    parser.add_argument(
        "--cpu_memory_gb",
        type=float,
        default=50.0,
        help="CPU memory limit for LMCache (GB)",
    )
    parser.add_argument(
        "--run_benchmark",
        action="store_true",
        help="Run full benchmark across multiple context lengths",
    )
    parser.add_argument(
        "--target_lens",
        type=str,
        default="4,8,16,32,64",
        help="Target context lengths in K (comma-separated)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output CSV file path (default: results/benchmark/vllm_lmcache_perf.csv)",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Maximum model length (default: auto, capped at 65536)",
    )
    parser.add_argument(
        "--no_cache_test",
        action="store_true",
        help="Disable warm cache testing (only test cold performance)",
    )
    args = parser.parse_args()

    use_lmcache = not args.no_lmcache
    test_cache_hit = not args.no_cache_test

    if args.run_benchmark:
        target_lens = [int(x) * 1024 for x in args.target_lens.split(",")]
        run_benchmark(
            args.model_name,
            args.tensor_parallel_size,
            use_lmcache,
            args.cpu_memory_gb,
            target_lens,
            args.output_file,
            args.max_model_len,
            test_cache_hit,
        )
    else:
        # Single run mode
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=True
        )
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1)

        max_len = min(args.context_window + 10_000, 196880)
        llm = create_llm(
            args.model_name,
            max_len,
            args.tensor_parallel_size,
            use_lmcache,
            args.cpu_memory_gb,
        )

        if args.attn_type == "minference":
            minference_patch = MInference("vllm_minference", args.model_name)
            llm = minference_patch(llm)

        run_target_length(
            args.context_window,
            llm,
            tokenizer,
            sampling_params,
            args.attn_type or "default",
            test_cache_hit=test_cache_hit and use_lmcache and HAS_LMCACHE,
        )
