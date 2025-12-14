# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""Save KV cache from model inference for microbenchmarking."""

import os, pickle, torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# Configuration
MODEL_PATH = "/home/zijie/models/Qwen3-0.6B/"
SEQ_LEN = 4096
LAYER_TO_SAVE = 0
CHUNK_SIZE = 2048
SAVE_DIR = "results/kvcache"

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model.eval()

    # Generate input and run prefill
    input_ids = torch.randint(100, 5000, (1, SEQ_LEN), dtype=torch.long)
    past_kv = DynamicCache()

    with torch.no_grad():
        for i in tqdm(range(0, SEQ_LEN, CHUNK_SIZE), desc="Prefilling"):
            output = model(input_ids=input_ids[:, i:i+CHUNK_SIZE].to(model.device), past_key_values=past_kv, use_cache=True)
            past_kv = output.past_key_values

    # Extract and save KV cache from target layer
    k, v = past_kv.layers[LAYER_TO_SAVE].keys, past_kv.layers[LAYER_TO_SAVE].values
    print(f"\n{'='*50}\nExtracted KV Cache (layer {LAYER_TO_SAVE}, seq_len {SEQ_LEN}):\n{'='*50}")
    print(f"key  : shape={k.shape}, dtype={k.dtype}")
    print(f"value: shape={v.shape}, dtype={v.dtype}")

    # Save to disk
    with open(os.path.join(SAVE_DIR, f"key_{SEQ_LEN}.pkl"), "wb") as f: pickle.dump(k.cpu(), f)
    with open(os.path.join(SAVE_DIR, f"value_{SEQ_LEN}.pkl"), "wb") as f: pickle.dump(v.cpu(), f)
    print(f"\nSaved to {SAVE_DIR}/")
