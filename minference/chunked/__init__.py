# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Chunked prefill module for MInference sparse attention profiling.

This module enables sparse attention profiling on extremely long sequences (1M+ tokens)
with limited GPU memory by using:
1. CPU offloading for K, V tensors
2. Chunked probe attention for pattern discovery
3. Online softmax for chunked sparse attention computation

Usage:
    from minference.chunked import ChunkedMInferenceProfiler

    profiler = ChunkedMInferenceProfiler(
        model_path="/path/to/model",
        seq_len=1_000_000,
    )
    results = profiler.run()
"""

from .profiler import ChunkedMInferenceProfiler
from .attention import (
    chunked_probe_attention,
    chunked_sparse_attention,
    online_softmax_update,
    minference_sparse_attention,
    chunked_minference_sparse_attention,
)
from .pattern_discovery import (
    sum_all_diagonal_matrix,
    extract_patterns,
)
from .cpu_offload import ChunkedKVCache

__all__ = [
    "ChunkedMInferenceProfiler",
    "chunked_probe_attention",
    "chunked_sparse_attention",
    "online_softmax_update",
    "minference_sparse_attention",
    "chunked_minference_sparse_attention",
    "sum_all_diagonal_matrix",
    "extract_patterns",
    "ChunkedKVCache",
]
