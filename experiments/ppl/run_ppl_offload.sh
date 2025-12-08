#!/bin/bash
# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# PPL evaluation with chunked prefill
# This enables evaluation on very long sequences with limited GPU memory
#
# Three modes available:
# - full: Keep full KV cache (most accurate, higher memory)
# - window: Sliding window KV cache (balanced memory/accuracy)
# - nocache: No KV cache (lowest memory, local attention only)

# Configuration - can be overridden at the command line
MODEL_NAME=${1:-/home/zijie/models/glm-4-9b-chat-1m}  # Model to evaluate
MIN_SEQ_LENGTH=${2:-32768}                             # Min sequence length
MAX_SEQ_LENGTH=${3:-10000000}                            # Max sequence length
CHUNK_SIZE=${4:-4096}                                 # Chunk size for forward pass
WINDOW_SIZE=${5:-32768}                               # KV cache window size
INTERVALS=${6:-10}                                     # Number of intervals
NUM_EVAL_EXAMPLES=${7:-20}                             # Number of examples to evaluate
MODE=${8:-full}                                       # Cache mode: full, window, nocache
OUTPUT_PATH=${9:-results/long-ppl-offload/}           # Output directory

export TOKENIZERS_PARALLELISM=false

echo "=== Chunked PPL Evaluation ==="
echo "Model: ${MODEL_NAME}"
echo "Min Seq Length: ${MIN_SEQ_LENGTH}"
echo "Max Seq Length: ${MAX_SEQ_LENGTH}"
echo "Chunk Size: ${CHUNK_SIZE}"
echo "Window Size: ${WINDOW_SIZE}"
echo "Mode: ${MODE}"
echo "Intervals: ${INTERVALS}"
echo "Eval Examples: ${NUM_EVAL_EXAMPLES}"
echo "Output Path: ${OUTPUT_PATH}"
echo ""

mkdir -p "${OUTPUT_PATH}"

python experiments/ppl/run_ppl_offload.py \
    --model_name "${MODEL_NAME}" \
    --min_seq_length "${MIN_SEQ_LENGTH}" \
    --max_seq_length "${MAX_SEQ_LENGTH}" \
    --chunk_size "${CHUNK_SIZE}" \
    --window_size "${WINDOW_SIZE}" \
    --intervals "${INTERVALS}" \
    --num_eval_examples "${NUM_EVAL_EXAMPLES}" \
    --mode "${MODE}" \
    --output_path "${OUTPUT_PATH}"

echo "Chunked PPL evaluation completed!"
echo "Results saved to: ${OUTPUT_PATH}"

# Usage examples:
#
# Basic usage (GLM-4 with 1K-32K, full context):
#   bash run_ppl_offload.sh
#
# Custom range 8K-64K:
#   bash run_ppl_offload.sh /home/zijie/models/glm-4-9b-chat-1m 8000 65536
#
# Llama-3 with 1K-100K, sliding window:
#   bash run_ppl_offload.sh /home/zijie/models/Llama-3-8B-Instruct-262k 1000 100000 4096 32768 9 5 window
#
# Very long sequences (>100K) with nocache mode:
#   bash run_ppl_offload.sh /path/to/model 1000 200000 8192 0 9 5 nocache
#
# Memory requirements per mode:
# - full: ~16GB model + O(seq_len) KV cache
# - window: ~16GB model + O(window_size) KV cache
# - nocache: ~16GB model + O(chunk_size) temporary
