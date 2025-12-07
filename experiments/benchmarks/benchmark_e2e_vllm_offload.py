# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Benchmark for MInference + vLLM V1 with LMCache KV cache offload.

This benchmark measures prefill latency with:
- MInference sparse attention
- LMCache CPU offload for prefix caching
- Tensor parallelism

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python experiments/benchmarks/benchmark_e2e_vllm_offload.py \
        --run_benchmark --model_name /path/to/model
"""

import argparse
import os
import time
from pathlib import Path

# LMCache configuration (must be set before importing vllm)
os.environ["LMCACHE_CHUNK_SIZE"] = "256"
os.environ["LMCACHE_LOCAL_CPU"] = "True"
os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"

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
):
    """Run benchmark for a specific context length."""
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

    s = 0
    T = num_iterations
    for i in range(T + 1):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            outputs = llm.generate([prompt], sampling_params)
        torch.cuda.synchronize()
        if i:  # skip warmup
            s += time.time() - start

    avg_time = s / T
    print(f"{attn_type} | {m} tokens | {avg_time:.4f}s")
    return avg_time


def create_llm(
    model_name: str,
    max_len: int,
    tensor_parallel_size: int,
    use_lmcache: bool,
    cpu_memory_gb: float,
):
    """Create LLM instance with optional LMCache offload."""
    llm_kwargs = {
        "model": model_name,
        "enforce_eager": True,
        "max_model_len": max_len,
        "enable_chunked_prefill": False,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": 0.85,
    }

    # Add KV cache offload if LMCache is available
    if use_lmcache and HAS_LMCACHE:
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(cpu_memory_gb)
        try:
            kv_config = KVTransferConfig(
                kv_connector="LMCacheConnectorV1",
                kv_role="kv_both",
            )
            llm_kwargs["kv_transfer_config"] = kv_config
            print(f"LMCache enabled with {cpu_memory_gb}GB CPU memory")
        except Exception as e:
            print(f"Warning: Could not configure LMCache: {e}")

    return LLM(**llm_kwargs)


def run_benchmark(
    model_name: str,
    tensor_parallel_size: int,
    use_lmcache: bool,
    cpu_memory_gb: float,
    target_lens: list = None,
    output_file: str = None,
    max_model_len: int = None,
):
    """Run full benchmark across different context lengths."""
    if target_lens is None:
        target_lens = [l * 1024 for l in [4, 8, 16, 32, 64]]

    ATTN_TYPES = ["flash_attn", "minference"]
    ATTN_TYPES2NAME = {
        "flash_attn": "FlashAttention-2",
        "minference": "MInference",
    }

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
                results.append(
                    {
                        "context_length": context_len,
                        "context_length_k": f"{context_len // 1024}K",
                        "attn_type": ATTN_TYPES2NAME[attn_type],
                        "latency_s": None,
                        "throughput_tokens_per_s": None,
                    }
                )
                continue

            latency = run_target_length(
                context_len, llm, tokenizer, sampling_params, attn_type
            )
            throughput = context_len / latency if latency > 0 else 0

            results.append(
                {
                    "context_length": context_len,
                    "context_length_k": f"{context_len // 1024}K",
                    "attn_type": ATTN_TYPES2NAME[attn_type],
                    "latency_s": latency,
                    "throughput_tokens_per_s": throughput,
                }
            )
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
    pivot_latency = df.pivot(
        index="context_length_k", columns="attn_type", values="latency_s"
    )
    pivot_throughput = df.pivot(
        index="context_length_k", columns="attn_type", values="throughput_tokens_per_s"
    )

    print("\nLatency (seconds):")
    print(pivot_latency.to_string())

    print("\nThroughput (tokens/s):")
    print(pivot_throughput.to_string())

    # Calculate speedup
    if "FlashAttention-2" in pivot_latency.columns and "MInference" in pivot_latency.columns:
        speedup = pivot_latency["FlashAttention-2"] / pivot_latency["MInference"]
        print("\nSpeedup (FlashAttention-2 / MInference):")
        print(speedup.to_string())

    # Save results
    if output_file is None:
        output_file = RESULTS_DIR / "vllm_offload_perf.csv"
    else:
        output_file = Path(output_file)

    # Ensure directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # Also save pivot tables
    pivot_file = output_file.parent / "vllm_offload_perf_pivot.csv"
    pivot_latency.to_csv(pivot_file)
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
        help="Output CSV file path (default: results/benchmark/vllm_offload_perf.csv)",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Maximum model length (default: auto, capped at 65536)",
    )
    args = parser.parse_args()

    use_lmcache = not args.no_lmcache

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
        )
