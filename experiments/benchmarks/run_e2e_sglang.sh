#!/bin/bash
# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# SGLang benchmark for MInference sparse attention
#
# SGLang integrates MInference through the `dual_chunk_flash_attn` attention backend.
# This requires models with built-in dual_chunk_attention_config in their config.json.
#
# Supported models:
#   - Qwen/Qwen2.5-7B-Instruct-1M
#   - Qwen/Qwen2.5-14B-Instruct-1M
#   - Other models with dual_chunk_attention_config

# Load data (download only if not exists)
[ -f prompt_hardest.txt ] || wget https://raw.githubusercontent.com/FranxYao/chain-of-thought-hub/main/gsm8k/lib_prompt/prompt_hardest.txt

# Use Qwen2.5-7B-Instruct-1M which has built-in dual_chunk_attention_config
MODEL="Qwen/Qwen2.5-7B-Instruct"

python experiments/benchmarks/benchmark_e2e_sglang.py --run_benchmark --model_name $MODEL