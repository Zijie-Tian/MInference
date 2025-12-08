#!/bin/bash
# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# =============================================================================
# MInference + vLLM V1 + LMCache Offload Benchmark
# =============================================================================
#
# This script benchmarks MInference with vLLM V1 engine and LMCache CPU offload.
#
# What LMCache does:
#   - Chunk-based KV cache offloading (256 tokens per chunk)
#   - GPU -> CPU -> Disk multi-level caching
#   - Cross-request prefix cache reuse
#   - Async offload without blocking inference
#
# Requirements:
#   - 2 GPUs (CUDA_VISIBLE_DEVICES=0,1)
#   - LMCache installed: pip install lmcache
#   - Model: Llama-3-8B-Instruct-262k (or similar long-context model)
#
# Usage:
#   bash experiments/benchmarks/run_e2e_vllm_offload.sh
#
# =============================================================================

set -e

# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1

# vLLM V1 configuration
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# IMPORTANT: PYTHONHASHSEED must be set for LMCache hash consistency
# Without this, cache hits won't work across processes
export PYTHONHASHSEED=0

# Load data (download only if not exists)
cd "$(dirname "$0")"
[ -f prompt_hardest.txt ] || wget https://raw.githubusercontent.com/FranxYao/chain-of-thought-hub/main/gsm8k/lib_prompt/prompt_hardest.txt

# Model configuration
MODEL="/home/zijie/models/Llama-3-8B-Instruct-262k"

echo "============================================================"
echo "MInference + vLLM V1 + LMCache Offload Benchmark"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  Model: $MODEL"
echo "  Tensor Parallel Size: 2"
echo "  LMCache: enabled (50GB CPU memory)"
echo ""
echo "This benchmark tests:"
echo "  1. Cold cache latency (first request, no cache)"
echo "  2. Warm cache latency (same prefix, cache hit)"
echo "  3. MInference speedup over FlashAttention-2"
echo ""
echo "============================================================"
echo ""

# Run benchmark
# Note: target_lens should not exceed max_model_len - 1000
# Default max_model_len is capped at 65536 for 2x 24GB GPUs
python benchmark_e2e_vllm_offload.py \
    --run_benchmark \
    --model_name "$MODEL" \
    --tensor_parallel_size 2 \
    --cpu_memory_gb 50.0 \
    --target_lens "4,8,16,32,64" \
    --max_model_len 65536

echo ""
echo "============================================================"
echo "Benchmark completed!"
echo "Results saved to results/benchmark/vllm_lmcache_perf.csv"
echo "============================================================"
