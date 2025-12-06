#!/bin/bash
# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# Configuration - can be overridden at the command line
MODEL=${1:-/home/zijie/models/Llama-3-8B-Instruct-262k}  # Model path or name
MAX_LENGTH=${2:-16000}                          # Max sequence length
NUM_EVAL=${3:--1}                                # Number of examples (-1=all)
ATTN_TYPE=${4:-minference}                       # Attention type: minference/sparq/hf/vllm

TASKS=("kv_retrieval" "longbook_choice_eng" "math_find" "longbook_qa_chn" "longbook_qa_eng" "longdialogue_qa_eng" "code_debug" "longbook_sum_eng" "number_string" "passkey")

echo "Model: ${MODEL}"
echo "Max Length: ${MAX_LENGTH}"
echo "Eval Examples: ${NUM_EVAL}"
echo "Attention Type: ${ATTN_TYPE}"
echo ""

export TOKENIZERS_PARALLELISM=false
SCRIPT_DIR=$(dirname "$0")

for task in ${TASKS[@]}; do
echo "Evaluating: ${task}"
python "$SCRIPT_DIR/run_infinitebench.py" \
    --task $task \
    --model_name_or_path "${MODEL}" \
    --data_dir ./data \
    --output_dir ./results \
    --max_seq_length ${MAX_LENGTH} \
    --rewrite \
    --num_eval_examples ${NUM_EVAL} \
    --topk 1 \
    --starting_layer 0 \
    --attn_type "${ATTN_TYPE}"
done

echo ""
echo "All tasks completed!"
echo "Results: ./results"

# Usage: bash run_infinitebench.sh [model] [max_length] [num_eval] [attn_type]
# Example: bash run_infinitebench.sh gradientai/Llama-3-8B-Instruct-262k 160000 -1 minference
