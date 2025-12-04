#!/bin/bash
# MInference Sparse Pattern Search Script
# Usage: ./tools/search_sparse_config.sh

set -e

#----------------------- Configuration -----------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_PATH="${PROJECT_ROOT}/tools/Qwen2.5_0.5B_Instruct_sparse_config.json"
SEQ_LENGTH=8192  # Reduced from 32768 to avoid OOM during profiling
DTYPE="bfloat16"
ATTN_IMPL="sdpa"  # or "flash_attention_2"
# Profile settings (optional, comment out to disable)
# PROFILE_DIR="${PROJECT_ROOT}/tools/profiles"
PROFILE_LAYERS=""  # e.g., "0,5,10" for specific layers, empty for all
#-------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "MInference Sparse Pattern Search"
echo "=============================================="
echo "Model:       $MODEL_NAME"
echo "Output:      $OUTPUT_PATH"
echo "Seq Length:  $SEQ_LENGTH"
echo "Dtype:       $DTYPE"
echo "Attn Impl:   $ATTN_IMPL"
echo "Profile Dir: $PROFILE_DIR"
echo "=============================================="

mkdir -p "$(dirname "$OUTPUT_PATH")"

# Build command with optional profile arguments
CMD="python ${SCRIPT_DIR}/search_sparse_config.py \
    --model_name $MODEL_NAME \
    --output_path $OUTPUT_PATH \
    --seq_length $SEQ_LENGTH \
    --dtype $DTYPE \
    --attn_impl $ATTN_IMPL \
    --trust_remote_code"

if [ -n "$PROFILE_DIR" ]; then
    CMD="$CMD --profile_dir $PROFILE_DIR"
fi
if [ -n "$PROFILE_LAYERS" ]; then
    CMD="$CMD --profile_layers $PROFILE_LAYERS"
fi

eval $CMD

echo ""
echo "Done! Config saved to: $OUTPUT_PATH"
if [ -n "$PROFILE_DIR" ]; then
    echo "Profile data saved to: $PROFILE_DIR"
fi
