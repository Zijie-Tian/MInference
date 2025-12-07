# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Long-context inference example with MInference + vLLM V1 + LMCache CPU offload.

This example demonstrates:
- MInference sparse attention for accelerated long-context inference
- vLLM V1 engine with tensor parallelism
- LMCache for KV cache CPU offload (prefix caching)

Usage:
    # Default 50K tokens
    CUDA_VISIBLE_DEVICES=0,1 python examples/run_vllm_long_context.py

    # Custom context length (e.g., 100K tokens)
    CUDA_VISIBLE_DEVICES=0,1 python examples/run_vllm_long_context.py --context-length 100000

    # Without LMCache (saves memory but no prefix caching)
    CUDA_VISIBLE_DEVICES=0,1 python examples/run_vllm_long_context.py --no-lmcache

    # Custom model
    CUDA_VISIBLE_DEVICES=0,1 python examples/run_vllm_long_context.py --model /path/to/model

Note:
    - Maximum context length is limited by GPU memory (~196K tokens with 2x 24GB GPUs)
    - LMCache enables KV cache prefix sharing between requests
"""

import argparse
import os
import sys

# vLLM configuration for MInference (must be set before importing vllm)
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

# LMCache configuration for KV cache offload to CPU
os.environ["LMCACHE_CHUNK_SIZE"] = "256"
os.environ["LMCACHE_LOCAL_CPU"] = "True"
os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Long-context inference with MInference + vLLM V1"
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=50000,
        help="Target context length in tokens (default: 50000)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/home/zijie/models/Llama-3-8B-Instruct-262k",
        help="Path to the model",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model length (default: auto-detect based on GPU memory)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Number of GPUs for tensor parallelism (default: 2)",
    )
    parser.add_argument(
        "--no-lmcache",
        action="store_true",
        help="Disable LMCache CPU offload",
    )
    parser.add_argument(
        "--cpu-memory-gb",
        type=float,
        default=50.0,
        help="CPU memory limit for LMCache in GB (default: 50.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)",
    )
    return parser.parse_args()


def generate_long_context(target_tokens: int = 50000) -> str:
    """
    Generate a long context prompt for needle-in-haystack testing.

    Args:
        target_tokens: Approximate number of tokens to generate

    Returns:
        A long context string with embedded information and Llama-3 chat format
    """
    needle = """
[IMPORTANT SECRET INFORMATION]
The secret code is: MINFERENCE-VLLM-LONGCONTEXT
The password for the vault is: SparseAttentionWorks
[END SECRET INFORMATION]
"""

    haystack_paragraphs = [
        "The development of artificial intelligence has been one of the most significant technological advances of the 21st century. Machine learning algorithms have become increasingly sophisticated, enabling computers to perform tasks that were once thought to be exclusively within the domain of human intelligence.",
        "Natural language processing has made remarkable progress in recent years. Models can now understand context, generate coherent text, and engage in meaningful conversations. This has led to the development of various applications, from chatbots to content generation tools.",
        "The field of computer vision has also seen tremendous growth. Deep learning models can now recognize objects, faces, and scenes with remarkable accuracy. This technology is being applied in autonomous vehicles, medical imaging, and security systems.",
        "Reinforcement learning has enabled machines to learn complex behaviors through trial and error. This approach has been successfully applied to game playing, robotics, and optimization problems.",
        "Cloud computing has democratized access to powerful computational resources. Organizations of all sizes can now leverage machine learning capabilities without investing in expensive hardware.",
        "Edge computing is emerging as a complement to cloud-based AI. By processing data closer to where it is generated, edge AI can reduce latency and improve privacy.",
        "Transfer learning has made it possible to leverage pre-trained models for new tasks. This approach significantly reduces the amount of data and computational resources required for training.",
        "Attention mechanisms have revolutionized the field of deep learning. The transformer architecture has become the foundation for state-of-the-art models in NLP and beyond.",
        "Sparse attention patterns have been identified as a way to reduce the computational cost of transformer models. By focusing on the most relevant parts of the input, these methods can achieve significant speedups.",
        "The ethical implications of AI are becoming increasingly important. Questions about bias, privacy, and the impact on employment are being actively discussed.",
    ]

    # Approximate: 1 token ~ 4 characters
    chars_per_token = 4
    target_chars = target_tokens * chars_per_token

    haystack = ""
    paragraph_idx = 0
    while len(haystack) < target_chars:
        haystack += haystack_paragraphs[paragraph_idx % len(haystack_paragraphs)]
        haystack += "\n\n"
        paragraph_idx += 1

    # Insert needle in the middle
    middle = len(haystack) // 2
    full_context = haystack[:middle] + needle + haystack[middle:]

    question = """

Based on the text above, please answer the following questions:
1. What is the secret code mentioned in the text?
2. What is the password for the vault?

Please provide your answers clearly."""

    user_content = full_context + question

    # Llama-3 Instruct chat template
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant. Read the provided text carefully and answer the questions accurately.<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt


def main():
    args = parse_args()

    # Set LMCache CPU memory limit
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = str(args.cpu_memory_gb)

    # Import after setting environment variables
    from vllm import LLM, SamplingParams

    from minference import MInference

    # Check LMCache availability
    use_lmcache = not args.no_lmcache
    if use_lmcache:
        try:
            from vllm.config import KVTransferConfig

            import lmcache

            HAS_LMCACHE = True
        except ImportError:
            print("Warning: LMCache not available, running without CPU offload")
            HAS_LMCACHE = False
            use_lmcache = False
    else:
        HAS_LMCACHE = False

    print("=" * 70)
    print("MInference + vLLM V1 Long Context Example")
    print("=" * 70)

    # Check model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Target context length: {args.context_length:,} tokens")
    print(f"  Tensor parallel size: {args.tensor_parallel_size} GPUs")
    print(f"  LMCache enabled: {use_lmcache}")
    if use_lmcache:
        print(f"  CPU memory limit: {args.cpu_memory_gb} GB")

    # Generate prompt
    print("\nGenerating long context prompt...")
    prompt = generate_long_context(target_tokens=args.context_length)
    approx_tokens = len(prompt) // 4
    print(f"Prompt length: {len(prompt):,} characters (~{approx_tokens:,} tokens)")

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        max_tokens=args.max_tokens,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )

    # Determine max_model_len
    max_model_len = args.max_model_len
    if max_model_len is None:
        # Auto-detect: use context length + some buffer, capped at reasonable limit
        max_model_len = min(args.context_length + 10000, 196880)
    print(f"  Max model length: {max_model_len:,}")

    # Initialize vLLM
    print("\nInitializing vLLM...")

    llm_kwargs = {
        "model": args.model,
        "max_num_seqs": 1,
        "enforce_eager": True,
        "max_model_len": max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": 0.85,
    }

    # Add KV cache offload if LMCache is available
    if use_lmcache and HAS_LMCACHE:
        try:
            kv_config = KVTransferConfig(
                kv_connector="LMCacheConnectorV1",
                kv_role="kv_both",
            )
            llm_kwargs["kv_transfer_config"] = kv_config
            print("  KV cache offload: enabled (LMCache)")
        except Exception as e:
            print(f"Warning: Could not configure KV transfer: {e}")

    llm = LLM(**llm_kwargs)

    # Apply MInference patch (vllm_minference for V1 compatibility)
    print("\nApplying MInference patch (vllm_minference)...")
    minference_patch = MInference("vllm_minference", args.model)
    llm = minference_patch(llm)

    # Generate
    print("\nGenerating response...")
    print("-" * 70)

    outputs = llm.generate([prompt], sampling_params)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"\nGenerated response:\n{generated_text}")

        # Verify needle-in-haystack
        print("\n" + "-" * 70)
        print("Needle-in-Haystack Verification:")
        if "MINFERENCE-VLLM-LONGCONTEXT" in generated_text:
            print("  [PASS] Secret code found correctly!")
        else:
            print("  [FAIL] Secret code not found in response")

        if "SparseAttentionWorks" in generated_text:
            print("  [PASS] Password found correctly!")
        else:
            print("  [FAIL] Password not found in response")

    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
