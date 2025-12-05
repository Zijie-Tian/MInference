# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# Load data (download only if not exists)
[ -f prompt_hardest.txt ] || wget https://raw.githubusercontent.com/FranxYao/chain-of-thought-hub/main/gsm8k/lib_prompt/prompt_hardest.txt

MODEL="Qwen/Qwen2.5-7B-Instruct"
VLLM_USE_V1=0 VLLM_ALLOW_INSECURE_SERIALIZATION=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
python experiments/benchmarks/benchmark_e2e_vllm_tp.py --run_benchmark --model_name $MODEL --tensor_parallel_size 4
