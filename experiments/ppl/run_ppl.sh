#!/bin/bash
# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# Configuration - can be overridden at the command line
MODEL_NAME=${1:-/home/zijie/models/glm-4-9b-chat-1m}  # Model to evaluate
# MODEL_NAME=${1:-/home/zijie/models/Llama-3-8B-Instruct-262k}  # Model to evaluate
ATTN_TYPE=${2:-minference}                             # Attention type
MAX_SEQ_LENGTH=${3:-32768}                           # Max sequence length
INTERVALS=${4:-19}                                    # Number of intervals
NUM_EVAL_EXAMPLES=${5:-5}                           # Number of examples to evaluate
OUTPUT_PATH=${6:-results/long-ppl/}                  # Output directory

export TOKENIZERS_PARALLELISM=false

echo "Model: ${MODEL_NAME}"
echo "Attention Type: ${ATTN_TYPE}"
echo "Max Seq Length: ${MAX_SEQ_LENGTH}"
echo "Intervals: ${INTERVALS}"
echo "Eval Examples: ${NUM_EVAL_EXAMPLES}"
echo "Output Path: ${OUTPUT_PATH}"
echo ""

mkdir -p "${OUTPUT_PATH}"
python experiments/ppl/run_ppl.py \
    --model_name "${MODEL_NAME}" \
    --attn_type "${ATTN_TYPE}" \
    --max_seq_length "${MAX_SEQ_LENGTH}" \
    --intervals "${INTERVALS}" \
    --num_eval_examples "${NUM_EVAL_EXAMPLES}" \
    --output_path "${OUTPUT_PATH}"

echo "PPL evaluation completed!"
echo "Results saved to: ${OUTPUT_PATH}"

# Usage: bash run_ppl.sh [model] [attn_type] [max_length] [intervals] [num_eval] [output_path]
# Example: bash run_ppl.sh gradientai/Llama-3-8B-Instruct-262k minference 100000 19 500 results/long-ppl/
