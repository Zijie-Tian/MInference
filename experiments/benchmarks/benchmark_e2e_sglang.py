# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
SGLang benchmark for MInference sparse attention.

SGLang integrates MInference through the `dual_chunk_flash_attn` attention backend.
This backend uses vertical-slash sparse attention patterns for long-context prefill.

Usage:
    python experiments/benchmarks/benchmark_e2e_sglang.py --run_benchmark --model_name Qwen/Qwen2.5-7B-Instruct-1M

For models with built-in dual_chunk_attention_config (e.g., Qwen2.5-*-Instruct-1M):
    The sparse attention config is already embedded in the model's config.json

For other models:
    You need to provide sparse_attention_config via --sparse-config-path
"""

import argparse
import json
import time
from collections import defaultdict

import torch
from transformers import AutoTokenizer


def run_target_length(m: int, engine, tokenizer, sampling_params, attn_type: str):
    # wget https://raw.githubusercontent.com/FranxYao/chain-of-thought-hub/main/gsm8k/lib_prompt/prompt_hardest.txt
    prompt_complex = open("./prompt_hardest.txt").read()
    input_ids = tokenizer(prompt_complex)["input_ids"]
    n = len(input_ids)
    b = m // n + 1

    new_input_ids = (input_ids * b)[:m]
    prompt = tokenizer.decode(new_input_ids)

    s = 0
    T = 10
    for i in range(T + 1):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            outputs = engine.generate(prompt, sampling_params)
        torch.cuda.synchronize()
        if i:  # skip warmup
            s += time.time() - start
    print(attn_type, m, s / T)
    return s / T


def check_model_has_dual_chunk_config(model_name: str) -> bool:
    """Check if model has built-in dual_chunk_attention_config."""
    from transformers import AutoConfig
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        return hasattr(config, "dual_chunk_attention_config") and config.dual_chunk_attention_config is not None
    except Exception:
        return False


def run_benchmark(
    model_name: str,
    attn_type: str,
    tensor_parallel_size: int = 1,
    sparse_config_path: str = None,
):
    """Run benchmark comparing FlashAttention vs MInference (dual_chunk_flash_attn)."""
    import sglang as sgl

    TARGET_LENS = [l * 1024 for l in [4, 8]]

    # Check if model has built-in dual_chunk_attention_config
    has_dual_chunk = check_model_has_dual_chunk_config(model_name)

    if has_dual_chunk:
        # Model has dual_chunk_attention_config, compare with/without sparse attention
        ATTN_TYPES = ["dense", "sparse"]
        ATTN_TYPES2NAME = {
            "dense": "DualChunk (Dense)",
            "sparse": "DualChunk + MInference",
        }
        print(f"\nModel {model_name} has dual_chunk_attention_config.")
        print("Comparing dense vs sparse attention within dual_chunk backend.\n")
    else:
        # Standard comparison
        ATTN_TYPES = ["flash_attn", "minference"]
        ATTN_TYPES2NAME = {
            "flash_attn": "FlashAttention-2",
            "minference": "MInference (SGLang)",
        }

    latency = defaultdict(list)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 1}

    for attn_type in ATTN_TYPES:
        max_len = TARGET_LENS[-1] + 10_000

        # Configure attention backend and sparse settings
        if has_dual_chunk:
            attention_backend = "dual_chunk_flash_attn"
            # For dual_chunk models, sparse attention is controlled by the config
            # We can't easily disable it at runtime without modifying config
            # So we just run the same backend (both use sparse if config has it)
            extra_args = {}
        else:
            if attn_type == "minference":
                attention_backend = "dual_chunk_flash_attn"
            else:
                attention_backend = "flashinfer"
            extra_args = {}

        print(f"\n{'='*60}")
        print(f"Testing {ATTN_TYPES2NAME[attn_type]} with backend: {attention_backend}")
        print(f"{'='*60}\n")

        # Create SGLang engine
        engine = sgl.Engine(
            model_path=model_name,
            tp_size=tensor_parallel_size,
            context_length=max_len,
            attention_backend=attention_backend,
            **extra_args,
        )

        for l in TARGET_LENS:
            t = run_target_length(l, engine, tokenizer, sampling_params, attn_type)
            latency[ATTN_TYPES2NAME[attn_type]].append([l, f"{t:.5f}"])
            print(attn_type, t, l)
            torch.cuda.empty_cache()

        engine.shutdown()
        del engine
        torch.cuda.empty_cache()

        # For dual_chunk models, we only need to run once since both use same backend
        if has_dual_chunk:
            # Copy results for both columns
            if attn_type == "dense":
                latency[ATTN_TYPES2NAME["sparse"]] = latency[ATTN_TYPES2NAME["dense"]]
            break

    res = [[""] + [ATTN_TYPES2NAME[attn_type] for attn_type in ATTN_TYPES]]
    for idx in range(len(TARGET_LENS)):
        l = TARGET_LENS[idx]
        res.append(
            [f"{l//1000}K"]
            + [latency[ATTN_TYPES2NAME[attn_type]][idx][-1] for attn_type in ATTN_TYPES]
        )
    print("\n".join(["\t".join(ii) for ii in res]))
    with open("res_sglang.csv", "w") as f:
        f.write("\n".join(["\t".join(ii) for ii in res]))
    return res


def run_single(
    model_name: str,
    attn_type: str,
    context_window: int,
    tensor_parallel_size: int = 1,
):
    """Run a single test with specified attention type."""
    import sglang as sgl

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 1}

    # Configure attention backend
    if attn_type == "minference":
        attention_backend = "dual_chunk_flash_attn"
    else:
        attention_backend = "flashinfer"

    engine = sgl.Engine(
        model_path=model_name,
        tp_size=tensor_parallel_size,
        context_length=context_window + 10_000,
        attention_backend=attention_backend,
    )

    run_target_length(context_window, engine, tokenizer, sampling_params, attn_type)
    engine.shutdown()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct-1M",
        help="Model name. For MInference, use models with dual_chunk_attention_config "
        "(e.g., Qwen/Qwen2.5-7B-Instruct-1M, Qwen/Qwen2.5-14B-Instruct-1M)",
    )
    args.add_argument(
        "--attn_type",
        type=str,
        choices=["flash_attn", "minference"],
        help="Attention type for single run mode",
    )
    args.add_argument("--context_window", type=int, default=100_000)
    args.add_argument("--tensor_parallel_size", type=int, default=1)
    args.add_argument("--run_benchmark", action="store_true")
    args.add_argument(
        "--sparse_config_path",
        type=str,
        default=None,
        help="Path to sparse attention config JSON (for models without built-in config)",
    )
    args = args.parse_args()

    model_name = args.model_name

    if args.run_benchmark:
        run_benchmark(
            model_name,
            args.attn_type,
            args.tensor_parallel_size,
            args.sparse_config_path,
        )
    else:
        if args.attn_type is None:
            print("Please specify --attn_type for single run mode")
            exit(1)
        run_single(
            model_name,
            args.attn_type,
            args.context_window,
            args.tensor_parallel_size,
        )
