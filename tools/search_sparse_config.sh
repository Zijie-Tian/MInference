#!/bin/bash
# MInference Sparse Pattern Search Script
# Usage: ./tools/search_sparse_config.sh

set -e

#----------------------- Configuration -----------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_PATH="${PROJECT_ROOT}/tools/Qwen2.5_0.5B_Instruct_sparse_config.json"
SEQ_LENGTH=32768
DTYPE="bfloat16"
ATTN_IMPL="sdpa"  # or "flash_attention_2"
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
echo "=============================================="

mkdir -p "$(dirname "$OUTPUT_PATH")"

python "${SCRIPT_DIR}/search_sparse_config.py" \
    --model_name "$MODEL_NAME" \
    --output_path "$OUTPUT_PATH" \
    --seq_length "$SEQ_LENGTH" \
    --dtype "$DTYPE" \
    --attn_impl "$ATTN_IMPL" \
    --trust_remote_code

echo ""
echo "Done! Config saved to: $OUTPUT_PATH"
