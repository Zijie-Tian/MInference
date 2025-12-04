# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Offline Sparse Pattern Search Tool for MInference

This script generates sparse attention config files for new models by running
pattern search on representative long-context data.

Usage:
    python tools/search_sparse_config.py \
        --model_name "your-model-name" \
        --output_path "./your_model_config.json" \
        --seq_length 32768

Requirements:
    - GPU with sufficient memory for the model
    - flash-attn installed
    - Long enough sequence for pattern discovery (recommended >= 32K tokens)
"""

import argparse
import json
import os
import sys
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for minference imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minference import MInference


def generate_long_text(tokenizer, target_length: int, method: str = "repeat") -> str:
    """
    Generate long text for pattern search.

    Args:
        tokenizer: The tokenizer to use for length estimation
        target_length: Target sequence length in tokens
        method: Text generation method
            - "repeat": Repeat a base passage
            - "random": Random token sequences (less realistic)

    Returns:
        Generated text string
    """
    if method == "repeat":
        # Use a representative passage and repeat it
        base_passage = """
The attention mechanism in transformer models computes relationships between all pairs
of tokens in a sequence. For a sequence of length N, this results in O(N²) complexity
for both computation and memory. This quadratic scaling becomes a significant bottleneck
when processing long sequences, limiting the practical context length of large language models.

Recent research has discovered that attention patterns in trained language models exhibit
inherent sparsity. Not all token pairs require computation - many attention weights are
near zero and contribute minimally to the output. This observation motivates sparse
attention methods that selectively compute only the important attention entries.

MInference leverages this dynamic sparse nature of attention by identifying common
patterns: vertical lines (attending to globally important tokens like BOS), slash
patterns (attending to recent context), and block sparse patterns. By detecting these
patterns during prefill and using optimized sparse kernels, MInference achieves
significant speedups while maintaining model quality.

The key insight is that while the exact sparse pattern varies across inputs, the
structure of sparsity (which heads use which pattern types, approximate sparsity levels)
remains relatively stable for a given model. This allows offline pattern search to
determine optimal configurations that work well across diverse inputs.
"""
        # Estimate tokens per passage
        tokens_per_passage = len(tokenizer.encode(base_passage))
        repetitions = (target_length // tokens_per_passage) + 1
        long_text = base_passage * repetitions

        # Verify and trim if needed
        tokens = tokenizer.encode(long_text)
        if len(tokens) > target_length:
            long_text = tokenizer.decode(tokens[:target_length])

        return long_text

    elif method == "random":
        # Generate random but valid token sequences
        vocab_size = tokenizer.vocab_size
        # Use common token range to avoid special tokens
        random_ids = torch.randint(1000, min(vocab_size, 30000), (target_length,))
        return tokenizer.decode(random_ids)

    else:
        raise ValueError(f"Unknown text generation method: {method}")


def load_text_from_file(file_path: str) -> str:
    """Load text from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def search_sparse_config(
    model_name: str,
    output_path: str,
    seq_length: int = 32768,
    input_file: Optional[str] = None,
    text_method: str = "repeat",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    resume: bool = True,
    attn_impl: str = "flash_attention_2",
):
    """
    Search for optimal sparse attention patterns for a model.

    Args:
        model_name: HuggingFace model name or path
        output_path: Path to save the config JSON
        seq_length: Target sequence length for search
        input_file: Optional file containing input text
        text_method: Method for generating text if no input_file
        dtype: Model dtype ("auto", "float16", "bfloat16")
        trust_remote_code: Whether to trust remote code
        resume: Whether to resume from existing partial config
    """
    print(f"=" * 60)
    print(f"MInference Sparse Pattern Search")
    print(f"=" * 60)
    print(f"Model: {model_name}")
    print(f"Output: {output_path}")
    print(f"Sequence Length: {seq_length}")
    print(f"=" * 60)

    # Check for existing config
    if os.path.exists(output_path):
        if resume:
            with open(output_path, "r") as f:
                existing_config = json.load(f)
            print(f"Found existing config with {len(existing_config)} layers, will resume...")
        else:
            print(f"Warning: {output_path} exists and will be overwritten")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare input text
    print("\n[2/4] Preparing input text...")
    if input_file:
        print(f"Loading text from {input_file}")
        input_text = load_text_from_file(input_file)
    else:
        print(f"Generating text using '{text_method}' method...")
        input_text = generate_long_text(tokenizer, seq_length, method=text_method)

    # Tokenize and check length
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=seq_length)
    actual_length = inputs["input_ids"].shape[1]
    print(f"Input sequence length: {actual_length} tokens")

    if actual_length < 8192:
        print(f"Warning: Sequence length {actual_length} is short. Recommended >= 32K for reliable pattern search.")

    # Load model
    print("\n[3/4] Loading model...")
    torch_dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype, "auto")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_impl,
    )

    # Ensure model is on CUDA
    if not hasattr(model, "hf_device_map"):
        model = model.cuda()

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    print(f"Model loaded: {num_layers} layers, {num_heads} heads per layer")

    # Apply MInference with search mode
    print("\n[4/4] Running pattern search...")
    print(f"This will search patterns for all {num_layers} layers...")
    print("Progress will be printed for each head.\n")

    minference_patch = MInference(
        attn_type="minference",
        model_name=model_name,
        config_path=output_path,
        is_search=True,  # Enable search mode
    )
    model = minference_patch(model)

    # Run inference to trigger search
    # Ensure model is on CUDA after patching
    model_device = next(model.parameters()).device
    if model_device.type == "cpu":
        print("Warning: Model was moved to CPU after patching, moving back to CUDA...")
        model = model.cuda()
        model_device = next(model.parameters()).device
    inputs = inputs.to(model_device)
    print(f"Model device: {model_device}")
    start_time = time.time()

    try:
        with torch.no_grad():
            # Generate just 1 token to complete prefill (where search happens)
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
            )

        elapsed = time.time() - start_time
        print(f"\n" + "=" * 60)
        print(f"Search completed in {elapsed:.2f} seconds")
        print(f"Config saved to: {output_path}")
        print(f"=" * 60)

        # Verify and summarize the config
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                config = json.load(f)

            print(f"\nConfig Summary:")
            print(f"  Total layers: {len(config)}")

            # Count pattern types
            pattern_counts = {}
            total_heads = 0
            for layer_config in config:
                for head_id, head_config in layer_config.items():
                    pattern_type = head_config[0]
                    pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
                    total_heads += 1

            print(f"  Total heads: {total_heads}")
            print(f"  Pattern distribution:")
            for pattern_type, count in sorted(pattern_counts.items()):
                pct = 100.0 * count / total_heads
                print(f"    {pattern_type}: {count} ({pct:.1f}%)")

    except RuntimeError as e:
        if "Search completed" in str(e) or "Search already completed" in str(e):
            print(f"\n{e}")
            print("This means the config file is complete!")
        else:
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Search sparse attention patterns for MInference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search patterns for a new model
  python tools/search_sparse_config.py \\
      --model_name "meta-llama/Llama-2-7b-hf" \\
      --output_path "./configs/llama2_7b_config.json"

  # Use custom input text
  python tools/search_sparse_config.py \\
      --model_name "your-model" \\
      --output_path "./config.json" \\
      --input_file "./your_long_text.txt"

  # Specify sequence length and dtype
  python tools/search_sparse_config.py \\
      --model_name "your-model" \\
      --output_path "./config.json" \\
      --seq_length 65536 \\
      --dtype bfloat16
"""
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the config JSON file",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=32768,
        help="Target sequence length for pattern search (default: 32768)",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Optional file containing input text for search",
    )
    parser.add_argument(
        "--text_method",
        type=str,
        default="repeat",
        choices=["repeat", "random"],
        help="Method for generating text if no input_file (default: repeat)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype (default: auto)",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading model",
    )
    parser.add_argument(
        "--attn_impl",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention implementation (default: flash_attention_2)",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Don't resume from existing partial config",
    )

    args = parser.parse_args()

    search_sparse_config(
        model_name=args.model_name,
        output_path=args.output_path,
        seq_length=args.seq_length,
        input_file=args.input_file,
        text_method=args.text_method,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        resume=not args.no_resume,
        attn_impl=args.attn_impl,
    )


if __name__ == "__main__":
    main()
