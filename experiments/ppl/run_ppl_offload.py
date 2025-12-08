# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
PPL evaluation with chunked prefill and CPU offload.

This module enables PPL evaluation on long sequences by using:
1. Chunked forward pass to manage GPU memory
2. Sliding window attention for very long sequences
3. Memory-efficient PPL computation

Usage:
    python experiments/ppl/run_ppl_offload.py \
        --model_name /path/to/model \
        --max_seq_length 100000 \
        --chunk_size 4096
"""

import argparse
import gc
import json
import math
import os
from typing import Optional

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache


class ChunkedPPL:
    """
    PPL evaluator using chunked forward pass with sliding window.

    This class enables perplexity evaluation on sequences longer than
    what can fit in GPU memory by:
    1. Processing input in chunks
    2. Using sliding window for KV cache
    3. Computing logits and PPL in chunks
    """

    def __init__(
        self,
        model_name: str,
        min_context: int,
        max_context: int,
        intervals: int = 10,
        run_name: str = None,
        output_path: str = "results/long-ppl-offload/",
        data_path: str = "liyucheng/pg19-4k",
        num_eval_examples: int = 100,
        chunk_size: int = 4096,
        window_size: int = 16384,  # KV cache window size
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.dtype = torch.float16
        self.device = "cuda"

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Load model config
        print(f"Loading model config from {model_name}...")
        self.model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        self.num_layers = self.model_config.num_hidden_layers
        self.num_heads = self.model_config.num_attention_heads
        self.num_kv_heads = getattr(
            self.model_config, 'num_key_value_heads',
            getattr(self.model_config, 'multi_query_group_num', self.num_heads)
        )
        self.head_dim = getattr(
            self.model_config, 'kv_channels',
            self.model_config.hidden_size // self.num_heads
        )
        self.hidden_dim = self.model_config.hidden_size

        print(f"Model: {self.num_layers} layers, {self.num_heads} heads, "
              f"{self.num_kv_heads} KV heads, {self.head_dim} head_dim")
        print(f"Chunk size: {chunk_size}, Window size: {window_size}")

        # Prepare data
        self.prepare_data(
            data_path, min_context, max_context, intervals, num_eval_examples
        )

        # Load model
        self.load_model()

        # Setup output
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.output_path = os.path.join(
            output_path,
            f'{model_name.replace("/", "-")}_chunked_{run_name if run_name is not None else ""}.json',
        )
        self.results = {}
        if os.path.exists(self.output_path):
            with open(self.output_path, "r") as f:
                self.results = json.load(f)

    def load_model(self):
        """Load the model with automatic device mapping."""
        print(f"Loading model weights...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def prepare_data(
        self,
        data_path: str,
        min_context: int,
        max_context: int,
        intervals: int,
        num_eval_examples: int,
    ):
        """Prepare evaluation data."""
        def tok(x):
            # Use tokenize + convert_tokens_to_ids to avoid padding issues
            tokens = self.tokenizer.tokenize(x["text"])
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            return {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
            }

        def truncate(x, length=None):
            return {
                "input_ids": x["input_ids"][:length],
                "attention_mask": x["attention_mask"][:length],
            }

        all_lengths = [
            min_context + (max_context - min_context) // intervals * i
            for i in range(intervals + 1)
        ]

        ds = datasets.load_dataset(data_path, split="train")
        ds1k = ds.select(range(num_eval_examples))

        ds1k = ds1k.map(tok, remove_columns=ds.column_names)

        self.test_data = {
            length: ds1k.map(truncate, fn_kwargs={"length": length})
            for length in all_lengths
        }

    def chunked_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Perform chunked forward pass.

        Process input in chunks to manage GPU memory. Uses sliding window
        for KV cache to handle very long sequences.

        Args:
            input_ids: Input token IDs [1, seq_len]

        Returns:
            logits: [1, seq_len, vocab_size]
        """
        seq_len = input_ids.shape[1]
        all_logits = []

        # Use model's cache mechanism
        past_key_values = None
        position_offset = 0

        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            chunk_ids = input_ids[:, start:end].to(self.device)

            with torch.no_grad():
                # Create position_ids for this chunk
                position_ids = torch.arange(start, end, device=self.device).unsqueeze(0)

                # Forward pass
                outputs = self.model(
                    input_ids=chunk_ids,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=False,
                )

                # Store logits
                all_logits.append(outputs.logits.cpu())

                # Update KV cache with window truncation
                past_key_values = outputs.past_key_values

                # Truncate KV cache if it exceeds window size
                if past_key_values is not None:
                    cache_len = self._get_cache_length(past_key_values)
                    if cache_len > self.window_size:
                        past_key_values = self._truncate_cache(
                            past_key_values, cache_len - self.window_size
                        )

            del chunk_ids, outputs
            torch.cuda.empty_cache()

        return torch.cat(all_logits, dim=1)

    def _get_cache_length(self, past_key_values) -> int:
        """Get the sequence length in the KV cache."""
        if past_key_values is None:
            return 0
        if isinstance(past_key_values, DynamicCache):
            return past_key_values.get_seq_length()
        elif isinstance(past_key_values, tuple):
            # (key, value) for each layer
            if len(past_key_values) > 0 and past_key_values[0] is not None:
                if isinstance(past_key_values[0], tuple):
                    return past_key_values[0][0].shape[2]
                else:
                    return past_key_values[0].shape[2]
        return 0

    def _truncate_cache(self, past_key_values, start_pos: int):
        """Truncate KV cache to keep only positions from start_pos onwards."""
        if isinstance(past_key_values, DynamicCache):
            # DynamicCache doesn't have built-in truncation, so we need to do it manually
            new_cache = DynamicCache()
            for layer_idx in range(len(past_key_values.key_cache)):
                key = past_key_values.key_cache[layer_idx][:, :, start_pos:, :]
                value = past_key_values.value_cache[layer_idx][:, :, start_pos:, :]
                new_cache.update(key, value, layer_idx)
            return new_cache
        elif isinstance(past_key_values, tuple):
            # Tuple of (key, value) for each layer
            new_cache = []
            for layer_cache in past_key_values:
                if isinstance(layer_cache, tuple):
                    key, value = layer_cache
                    new_cache.append((
                        key[:, :, start_pos:, :],
                        value[:, :, start_pos:, :]
                    ))
                else:
                    new_cache.append(layer_cache[:, :, start_pos:, :])
            return tuple(new_cache)
        return past_key_values

    def chunk_ppl(self, logits: torch.Tensor, labels: torch.Tensor, chunk_size: int = 2048) -> float:
        """Calculate PPL in chunks to save memory."""
        total_loss = 0.0
        total_tokens = 0
        seq_len = logits.size(1)

        for i in range(0, seq_len - 1, chunk_size):
            end = min(i + chunk_size, seq_len - 1)
            chunk_logits = logits[:, i:end, :].contiguous()
            chunk_labels = labels[:, i + 1:end + 1].contiguous()

            chunk_prob = F.log_softmax(chunk_logits, dim=-1, dtype=torch.float32)
            chunk_labels_flat = chunk_labels.view(-1)
            chunk_prob_flat = chunk_prob.view(-1, chunk_prob.size(-1))

            target_log_probs = chunk_prob_flat[
                torch.arange(chunk_prob_flat.size(0), device=chunk_prob_flat.device),
                chunk_labels_flat
            ]

            chunk_loss = -target_log_probs.sum().item()
            total_loss += chunk_loss
            total_tokens += chunk_labels_flat.size(0)

            del chunk_logits, chunk_labels, chunk_prob, chunk_prob_flat, target_log_probs

        avg_loss = total_loss / total_tokens
        return np.exp(avg_loss)

    def save_results(self):
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

    def start_test(self):
        """Run PPL evaluation."""
        print("Starting chunked PPL test...")

        for length, ds in self.test_data.items():
            ppls = []
            with torch.no_grad():
                for example in tqdm(ds, desc=f"Testing with context length: {length}"):
                    gc.collect()
                    torch.cuda.empty_cache()

                    input_ids = torch.tensor([example["input_ids"]], dtype=torch.long)

                    # Use chunked forward pass
                    logits = self.chunked_forward(input_ids)

                    # Calculate PPL
                    ppl = self.chunk_ppl(logits, input_ids)
                    ppls.append(ppl)

                    del logits
                    gc.collect()
                    torch.cuda.empty_cache()

            length_key = f"{length // 1_000}K"
            self.results[length_key] = np.mean(ppls)
            print(f"Average PPL for {length_key}: {self.results[length_key]}")
            self.save_results()

        print("Completed.")


class ChunkedPPLWithFullContext(ChunkedPPL):
    """
    PPL evaluator using chunked forward pass with full context.

    This variant does NOT use sliding window - it keeps the full KV cache.
    This is more accurate but requires more memory.

    For very long sequences (>32K), use the base ChunkedPPL class with
    window_size parameter instead.
    """

    def chunked_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Perform chunked forward pass with full context KV cache.

        Args:
            input_ids: Input token IDs [1, seq_len]

        Returns:
            logits: [1, seq_len, vocab_size]
        """
        seq_len = input_ids.shape[1]
        all_logits = []

        # Use model's cache mechanism
        past_key_values = None

        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            chunk_ids = input_ids[:, start:end].to(self.device)

            with torch.no_grad():
                # Create position_ids for this chunk
                position_ids = torch.arange(start, end, device=self.device).unsqueeze(0)

                # Forward pass
                outputs = self.model(
                    input_ids=chunk_ids,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=False,
                )

                # Store logits
                all_logits.append(outputs.logits.cpu())

                # Keep full KV cache (no truncation)
                past_key_values = outputs.past_key_values

            del chunk_ids, outputs
            torch.cuda.empty_cache()

        return torch.cat(all_logits, dim=1)


class ChunkedPPLNoCache(ChunkedPPL):
    """
    PPL evaluator without KV cache - processes each position independently.

    This is the most memory-efficient but slowest method. Good for very
    long sequences where even sliding window doesn't fit.

    Note: This uses local attention only within each chunk, so accuracy
    may be lower for sequences that need long-range dependencies.
    """

    def chunked_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Perform chunked forward pass without KV cache.

        Each chunk is processed independently with causal attention
        only within the chunk.

        Args:
            input_ids: Input token IDs [1, seq_len]

        Returns:
            logits: [1, seq_len, vocab_size]
        """
        seq_len = input_ids.shape[1]
        all_logits = []

        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            chunk_ids = input_ids[:, start:end].to(self.device)

            with torch.no_grad():
                # Forward pass without cache - just process this chunk
                outputs = self.model(
                    input_ids=chunk_ids,
                    use_cache=False,
                    output_hidden_states=False,
                )

                # Store logits
                all_logits.append(outputs.logits.cpu())

            del chunk_ids, outputs
            torch.cuda.empty_cache()

        return torch.cat(all_logits, dim=1)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, required=True)
    args.add_argument("--min_seq_length", type=int, default=1_000)
    args.add_argument("--max_seq_length", type=int, default=100_000)
    args.add_argument("--intervals", type=int, default=9)
    args.add_argument("--run_name", type=str, default=None)
    args.add_argument("--num_eval_examples", type=int, default=5)
    args.add_argument("--output_path", type=str, default="results/long-ppl-offload/")
    args.add_argument("--chunk_size", type=int, default=4096,
                      help="Chunk size for forward pass")
    args.add_argument("--window_size", type=int, default=16384,
                      help="KV cache window size (set to 0 for full context)")
    args.add_argument("--mode", type=str, default="window",
                      choices=["window", "full", "nocache"],
                      help="Cache mode: window (sliding window), full (full context), nocache (no cache)")

    args = args.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Select PPL evaluator based on mode
    if args.mode == "full":
        PPLClass = ChunkedPPLWithFullContext
        print("Using full context KV cache (most accurate, high memory)")
    elif args.mode == "nocache":
        PPLClass = ChunkedPPLNoCache
        print("Using no cache mode (lowest memory, local attention only)")
    else:
        PPLClass = ChunkedPPL
        print(f"Using sliding window mode with window_size={args.window_size}")

    test = PPLClass(
        model_name=args.model_name,
        min_context=args.min_seq_length,
        max_context=args.max_seq_length,
        intervals=args.intervals,
        run_name=args.run_name,
        num_eval_examples=args.num_eval_examples,
        output_path=args.output_path,
        chunk_size=args.chunk_size,
        window_size=args.window_size,
    )
    test.start_test()
