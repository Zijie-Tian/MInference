# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# Load data
wget -nc https://raw.githubusercontent.com/FranxYao/chain-of-thought-hub/main/gsm8k/lib_prompt/prompt_hardest.txt

# vLLM 0.9.0+ requires these environment variables:
# - VLLM_USE_V1=0: Use V0 engine (V1 engine has incompatible FlashAttentionImpl)
# - VLLM_ALLOW_INSECURE_SERIALIZATION=1: Allow pickle serialization for collective_rpc
# - VLLM_WORKER_MULTIPROC_METHOD=spawn: Use spawn for multiprocessing
VLLM_USE_V1=0 \
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python experiments/benchmarks/benchmark_e2e_vllm_tp.py \
    --attn_type minference \
    --context_window 100_000 \
    --tensor_parallel_size 4
