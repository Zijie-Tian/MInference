#!/bin/bash
# Run 1M context sparse attention profiling with visualization
# Usage: CUDA_VISIBLE_DEVICES=4,5 bash tools/run_profile_analysis.sh [seq_len]

set -e

SEQ_LEN=${1:-1000000}
MODEL=${MODEL:-"/home/zijie/models/Llama-3-8B-Instruct-262k"}
SAVE_DIR=${SAVE_DIR:-"results/profile"}
VERTICAL_SIZE=${VERTICAL_SIZE:-1000}
SLASH_SIZE=${SLASH_SIZE:-6096}
NUM_LAYERS=${NUM_LAYERS:-32}
NUM_WORKERS=${NUM_WORKERS:-32}

echo "========================================"
echo "MInference 1M Context Profiling"
echo "========================================"
echo "Sequence Length: $SEQ_LEN"
echo "Model: $MODEL"
echo "Save Dir: $SAVE_DIR"
echo "========================================"

# Step 1: Run profiler
echo "[1/2] Running chunked profiler..."
python tools/profile_sparse_attention_1m.py \
    --model "$MODEL" \
    --seq-len "$SEQ_LEN" \
    --save-dir "$SAVE_DIR" \
    --vertical-size "$VERTICAL_SIZE" \
    --slash-size "$SLASH_SIZE"

RESULT_DIR="${SAVE_DIR}/${SEQ_LEN}"

# Step 2: Generate visualization for all layers (parallel)
echo "[2/2] Generating visualization for $NUM_LAYERS layers ($NUM_WORKERS workers)..."
seq 0 $((NUM_LAYERS - 1)) | xargs -P "$NUM_WORKERS" -I {} \
    python tools/visualize_layer_computation.py --profile-dir "$RESULT_DIR" --layer {}

echo "========================================"
echo "Done! Results saved to: $RESULT_DIR"
echo "  - profile.json, summary.npz"
echo "  - viz_bs64/layer_*.png"
echo "========================================"
