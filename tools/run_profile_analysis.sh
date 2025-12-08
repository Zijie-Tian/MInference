#!/bin/bash
# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# MInference Sparse Attention Profiling & Analysis Pipeline
#
# This script runs the complete profiling and visualization pipeline:
# 1. Profile sparse attention patterns during model inference
# 2. Generate summary visualization with GQA grouping
# 3. Generate per-layer block computation visualizations
#
# Usage:
#   ./tools/run_profile_analysis.sh                    # Default: 16K tokens
#   ./tools/run_profile_analysis.sh --seq-len 8000     # Custom sequence length
#   ./tools/run_profile_analysis.sh --visualize-only   # Skip profiling, only visualize
#
# Output:
#   results/profile/{seq_len}/
#   ├── profile.json          # Full profile data
#   ├── summary.npz           # Summary arrays
#   ├── profile_viz.png       # Heatmap overview
#   ├── layers/               # Per-layer NPZ data
#   │   ├── layer_00.npz
#   │   └── ...
#   └── layer_XX_blocks.png   # Per-layer block visualizations

set -e

# Default parameters
SEQ_LEN=16000
MODEL="/home/zijie/models/Llama-3-8B-Instruct-262k"
SAVE_DIR="results/profile"
VISUALIZE_ONLY=false
CUDA_DEVICES="0,1"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seq-len)
            SEQ_LEN="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --visualize-only)
            VISUALIZE_ONLY=true
            shift
            ;;
        --cuda)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --seq-len N        Sequence length for profiling (default: 16000)"
            echo "  --model PATH       Model path (default: Llama-3-8B-Instruct-262k)"
            echo "  --save-dir PATH    Output directory (default: results/profile)"
            echo "  --visualize-only   Skip profiling, only generate visualizations"
            echo "  --cuda DEVICES     CUDA devices (default: 0,1)"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "========================================================================"
echo "MInference Sparse Attention Profiling & Analysis"
echo "========================================================================"
echo "Sequence Length: $SEQ_LEN"
echo "Model: $MODEL"
echo "Save Directory: $SAVE_DIR"
echo "CUDA Devices: $CUDA_DEVICES"
echo "========================================================================"

PROFILE_DIR="$SAVE_DIR/$SEQ_LEN"

# Step 1: Run profiling (unless --visualize-only)
if [ "$VISUALIZE_ONLY" = false ]; then
    echo ""
    echo "[Step 1/3] Running sparse attention profiling..."
    echo "------------------------------------------------------------------------"

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python tools/profile_sparse_attention.py \
        --model "$MODEL" \
        --seq-len "$SEQ_LEN" \
        --save-dir "$SAVE_DIR"

    echo ""
    echo "Profiling complete."
else
    echo ""
    echo "[Step 1/3] Skipping profiling (--visualize-only mode)"

    if [ ! -d "$PROFILE_DIR" ]; then
        echo "Error: Profile directory not found: $PROFILE_DIR"
        echo "Run without --visualize-only to generate profile data first."
        exit 1
    fi
fi

# Step 2: Generate summary visualization
echo ""
echo "[Step 2/3] Generating summary visualization..."
echo "------------------------------------------------------------------------"

python -c "
import sys
sys.path.insert(0, 'tools')
from profile_sparse_attention import visualize_profile
visualize_profile('$PROFILE_DIR')
"

echo "Summary visualization saved."

# Step 3: Generate per-layer visualizations
echo ""
echo "[Step 3/3] Generating per-layer block visualizations..."
echo "------------------------------------------------------------------------"

# Get number of layers from profile
NUM_LAYERS=$(python -c "
import json
with open('$PROFILE_DIR/profile.json') as f:
    data = json.load(f)
print(data['num_layers'])
")

echo "Generating visualizations for $NUM_LAYERS layers..."

for i in $(seq 0 $((NUM_LAYERS - 1))); do
    python tools/visualize_layer_computation.py \
        --profile-dir "$PROFILE_DIR" \
        --layer "$i"
done

echo ""
echo "========================================================================"
echo "Analysis Complete!"
echo "========================================================================"
echo ""
echo "Output files:"
echo "  $PROFILE_DIR/"
echo "  ├── profile.json           # Full profile data"
echo "  ├── summary.npz            # Summary arrays"
echo "  ├── profile_viz.png        # Heatmap overview"
echo "  ├── layers/                # Per-layer NPZ data"
echo "  └── layer_XX_blocks.png    # Per-layer block visualizations"
echo ""
echo "Quick view commands:"
echo "  # View summary heatmap"
echo "  xdg-open $PROFILE_DIR/profile_viz.png"
echo ""
echo "  # View specific layer"
echo "  xdg-open $PROFILE_DIR/layer_00_blocks.png"
echo ""
