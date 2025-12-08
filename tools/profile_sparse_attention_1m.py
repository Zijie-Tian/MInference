#!/usr/bin/env python
# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Profile sparse attention patterns on 1M+ context sequences.

This script uses chunked prefill with CPU offloading to profile MInference
sparse attention patterns on extremely long sequences with limited GPU memory.

Usage:
    python tools/profile_sparse_attention_1m.py \
        --model /path/to/model \
        --seq-len 1000000 \
        --save-dir results/profile

Example:
    # Profile 1M tokens
    python tools/profile_sparse_attention_1m.py \
        --model /home/zijie/models/Llama-3-8B-Instruct-262k \
        --seq-len 1000000

    # Profile 100K tokens (faster)
    python tools/profile_sparse_attention_1m.py \
        --model /home/zijie/models/Llama-3-8B-Instruct-262k \
        --seq-len 100000
"""

import os
import sys
import argparse
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minference.chunked import ChunkedMInferenceProfiler
from minference.chunked.profiler import ProfilerConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile sparse attention patterns on long sequences"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="/home/zijie/models/Llama-3-8B-Instruct-262k",
        help="Path to the model"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1_000_000,
        help="Sequence length to profile (default: 1M)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/profile",
        help="Directory to save results"
    )
    parser.add_argument(
        "--vertical-size",
        type=int,
        default=1000,
        help="Number of vertical columns (default: 1000)"
    )
    parser.add_argument(
        "--slash-size",
        type=int,
        default=6096,
        help="Number of slash diagonals (default: 6096)"
    )
    parser.add_argument(
        "--q-chunk-size",
        type=int,
        default=512,
        help="Query chunk size (default: 512)"
    )
    parser.add_argument(
        "--kv-chunk-size",
        type=int,
        default=65536,
        help="KV chunk size (default: 65536)"
    )
    parser.add_argument(
        "--embedding-chunk-size",
        type=int,
        default=8192,
        help="Embedding chunk size (default: 8192)"
    )
    parser.add_argument(
        "--compute-output",
        action="store_true",
        help="Compute sparse attention output (slower)"
    )
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help="Save attention outputs to disk"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for computation"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Print configuration
    print("=" * 70)
    print("Chunked Sparse Attention Profiler")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Sequence length: {args.seq_len:,}")
    print(f"Vertical size: {args.vertical_size}")
    print(f"Slash size: {args.slash_size}")
    print(f"Q chunk size: {args.q_chunk_size}")
    print(f"KV chunk size: {args.kv_chunk_size}")
    print(f"Embedding chunk size: {args.embedding_chunk_size}")
    print(f"Compute output: {args.compute_output}")
    print(f"Save outputs: {args.save_outputs}")
    print(f"Dtype: {args.dtype}")
    print(f"Save directory: {args.save_dir}")
    print("=" * 70)

    # Map dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Create profiler config
    config = ProfilerConfig(
        model_path=args.model,
        seq_len=args.seq_len,
        vertical_size=args.vertical_size,
        slash_size=args.slash_size,
        q_chunk_size=args.q_chunk_size,
        kv_chunk_size=args.kv_chunk_size,
        embedding_chunk_size=args.embedding_chunk_size,
        save_dir=args.save_dir,
        compute_output=args.compute_output,
        save_outputs=args.save_outputs,
        dtype=dtype,
    )

    # Create and run profiler
    profiler = ChunkedMInferenceProfiler(config)
    results = profiler.run()

    # Print summary
    print("\n" + "=" * 70)
    print("Profiling Summary")
    print("=" * 70)
    print(f"Layers: {results['num_layers']}")
    print(f"Heads: {results['num_heads']}")
    print(f"Total time: {results['total_time_seconds']:.1f} seconds")

    # Compute average sparsity
    total_sparsity = 0
    count = 0
    for layer_idx in range(results['num_layers']):
        for head_idx in range(results['num_heads']):
            head_data = results['layers'][str(layer_idx)]['heads'][str(head_idx)]
            total_sparsity += head_data['sparsity_ratio']
            count += 1

    avg_sparsity = total_sparsity / count
    print(f"Average sparsity ratio: {avg_sparsity:.4f} ({avg_sparsity*100:.2f}%)")

    output_dir = os.path.join(args.save_dir, str(args.seq_len))
    print(f"\nResults saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
