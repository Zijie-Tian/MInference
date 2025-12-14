# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Save KV cache and benchmark sparse attention methods.
Compares MInference, FlexPrefill, and Flash Attention.
"""

import os, pickle, torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from minference.modules.minference_forward import search_pattern_v2, minference_prefill_kernel
from minference.modules.flexprefill import flex_prefill_attention

try:
    from minference.modules.xattention import Xattention_prefill
    HAS_XATTENTION = True
except ImportError:
    HAS_XATTENTION = False

try:
    from flash_attn import flash_attn_func
except ImportError:
    from minference.ops.flash_attn_triton import _flash_attn_triton_decoding as flash_attn_func

# Configuration
MODEL_PATH = "/home/zijie/models/Qwen3-0.6B/"
SEQ_LEN = 16384
LAYER_TO_SAVE = 0
CHUNK_SIZE = 2048
SAVE_DIR = "results/kvcache"
WARMUP = 10
REPEAT = 100

def benchmark_kernel(func, *args, warmup=10, repeat=100, **kwargs):
    """Benchmark a kernel function using CUDA events."""
    for _ in range(warmup):
        _ = func(*args, **kwargs)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times = torch.tensor(times)
    return times.mean().item(), times.std().item()


def minference_prefill_all_heads(q, k, v, config, layer_idx=0):
    """
    MInference prefill for all heads (similar to minference_prefill_forward).
    Input shape: [batch, num_heads, seq_len, head_dim]
    """
    bsz, num_heads, q_len, head_dim = q.shape
    output = torch.empty_like(q)

    for head in range(num_heads):
        q_h = q[:, head:head+1, :, :]
        k_h = k[:, head:head+1, :, :]
        v_h = v[:, head:head+1, :, :]
        attn_output = minference_prefill_kernel(q_h, k_h, v_h, head, layer_idx, config)
        output[:, head:head+1] = attn_output

    return output


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

    # Extract KV cache from target layer
    k, v = past_kv.layers[LAYER_TO_SAVE].keys, past_kv.layers[LAYER_TO_SAVE].values
    print(f"\n{'='*60}\nExtracted KV Cache (layer {LAYER_TO_SAVE}, seq_len {SEQ_LEN}):\n{'='*60}")
    print(f"key  : shape={k.shape}, dtype={k.dtype}")
    print(f"value: shape={v.shape}, dtype={v.dtype}")

    # Generate random query
    num_heads = k.shape[1]
    q = torch.randn(1, num_heads, SEQ_LEN, k.shape[-1], device=k.device, dtype=k.dtype)

    # ============================================================================
    # Section 1: Search optimal sparse pattern for each head
    # ============================================================================
    print(f"\n{'='*60}")
    print("Section 1: Searching sparse patterns")
    print(f"{'='*60}")

    best_patterns = {}  # {head_id: (type, v_size, s_size, score)}
    for head in range(num_heads):
        q_h = q[:, head:head+1, :, :]
        k_h = k[:, head:head+1, :, :]
        v_h = v[:, head:head+1, :, :]
        all_info = search_pattern_v2(q_h, k_h, v_h, head)
        # Find best pattern (lowest score)
        best = min(all_info, key=lambda x: x[3])
        best_patterns[str(head)] = tuple(best)

    # ============================================================================
    # Section 2: Benchmark MInference vs FlexPrefill vs Flash Attention
    # ============================================================================
    print(f"\n{'='*60}")
    print("Section 2: Benchmarking All Methods (all heads)")
    print(f"{'='*60}")

    # Build config for minference
    config = {"best_pattern": {LAYER_TO_SAVE: best_patterns}}

    # Prepare inputs for FlexPrefill (expects [batch, seq_len, num_heads, head_dim])
    q_bnhd = q.transpose(1, 2).contiguous()
    k_bnhd = k.transpose(1, 2).contiguous()
    v_bnhd = v.transpose(1, 2).contiguous()

    # FlexPrefill parameters
    GAMMA = 0.9
    TAU = 0.5
    BLOCK_SIZE = 32

    # XAttention parameters (block_size must be 128 for block_sparse_attn_func)
    XATTN_STRIDE = 8
    XATTN_THRESHOLD = 0.9
    XATTN_BLOCK_SIZE = 128

    print(f"\nConfig: seq_len={SEQ_LEN}, num_heads={num_heads}, warmup={WARMUP}, repeat={REPEAT}")
    print(f"FlexPrefill: gamma={GAMMA}, tau={TAU}, block_size={BLOCK_SIZE}")
    print(f"XAttention: stride={XATTN_STRIDE}, threshold={XATTN_THRESHOLD}, block_size={XATTN_BLOCK_SIZE}")

    # Benchmark Flash Attention (all heads)
    flash_time, flash_std = benchmark_kernel(
        flash_attn_func,
        q_bnhd, k_bnhd, v_bnhd, causal=True,
        warmup=WARMUP, repeat=REPEAT
    )

    # Benchmark MInference (all heads)
    minference_time, minference_std = benchmark_kernel(
        minference_prefill_all_heads,
        q, k, v, config, LAYER_TO_SAVE,
        warmup=WARMUP, repeat=REPEAT
    )

    # Benchmark FlexPrefill (all heads)
    flexprefill_time, flexprefill_std = benchmark_kernel(
        flex_prefill_attention,
        q_bnhd, k_bnhd, v_bnhd, GAMMA, TAU, block_size=BLOCK_SIZE,
        warmup=WARMUP, repeat=REPEAT
    )

    # Benchmark XAttention (all heads) - input shape: [batch, heads, seq, dim]
    if HAS_XATTENTION:
        try:
            xattn_time, xattn_std = benchmark_kernel(
                Xattention_prefill,
                q, k, v, XATTN_STRIDE,
                norm=1, threshold=XATTN_THRESHOLD, block_size=XATTN_BLOCK_SIZE,
                warmup=WARMUP, repeat=REPEAT
            )
        except Exception as e:
            print(f"XAttention error: {e}")
            xattn_time, xattn_std = float('inf'), 0
    else:
        xattn_time, xattn_std = float('inf'), 0

    # Print results
    print(f"\n{'Method':<20} {'Time (ms)':<20} {'vs Flash'}")
    print("-" * 55)
    print(f"{'Flash Attention':<20} {flash_time:.3f} ± {flash_std:.3f}         1.00x")
    print(f"{'MInference':<20} {minference_time:.3f} ± {minference_std:.3f}         {flash_time/minference_time:.2f}x")
    print(f"{'FlexPrefill':<20} {flexprefill_time:.3f} ± {flexprefill_std:.3f}         {flash_time/flexprefill_time:.2f}x")
    if HAS_XATTENTION and xattn_time != float('inf'):
        print(f"{'XAttention':<20} {xattn_time:.3f} ± {xattn_std:.3f}         {flash_time/xattn_time:.2f}x")
    else:
        print(f"{'XAttention':<20} {'N/A (requires block_sparse_attn)'}")

    # ============================================================================
    # Section 3: Save KV cache
    # ============================================================================
    print(f"\n{'='*60}")
    print("Section 3: Saving KV cache")
    print(f"{'='*60}")

    with open(os.path.join(SAVE_DIR, f"key_{SEQ_LEN}.pkl"), "wb") as f: pickle.dump(k.cpu(), f)
    with open(os.path.join(SAVE_DIR, f"value_{SEQ_LEN}.pkl"), "wb") as f: pickle.dump(v.cpu(), f)
    print(f"Saved to {SAVE_DIR}/")
