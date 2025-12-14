// Copyright (c) 2024-2025 Microsoft
// Licensed under The MIT License [see LICENSE for details]

#include <torch/extension.h>
#include "sparse_attention.hpp"

namespace cpu_sparse_attn {

// PyTorch wrapper for vertical_slash_attention_cpu
torch::Tensor vertical_slash_attention_forward(
    torch::Tensor Q,           // [batch, heads, seq_len, head_dim]
    torch::Tensor K,           // [batch, heads, seq_len, head_dim]
    torch::Tensor V,           // [batch, heads, seq_len, head_dim]
    int64_t vertical_size,
    int64_t slash_size
) {
    // Input validation
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D tensor");
    TORCH_CHECK(K.dim() == 4, "K must be 4D tensor");
    TORCH_CHECK(V.dim() == 4, "V must be 4D tensor");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "K must be float32");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "V must be float32");

    int batch_size = Q.size(0);
    int num_heads = Q.size(1);
    int seq_len = Q.size(2);
    int head_dim = Q.size(3);

    // Allocate output
    auto output = torch::zeros_like(Q);

    // Call CPU kernel
    vertical_slash_attention_cpu(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        static_cast<int>(vertical_size),
        static_cast<int>(slash_size)
    );

    return output;
}

// Estimate-only function (for debugging/analysis)
std::tuple<torch::Tensor, torch::Tensor> estimate_pattern_forward(
    torch::Tensor Q,           // [seq_len, head_dim]
    torch::Tensor K,           // [seq_len, head_dim]
    int64_t vertical_size,
    int64_t slash_size
) {
    TORCH_CHECK(Q.dim() == 2, "Q must be 2D tensor [seq_len, head_dim]");
    TORCH_CHECK(K.dim() == 2, "K must be 2D tensor [seq_len, head_dim]");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");

    int seq_len = Q.size(0);
    int head_dim = Q.size(1);

    // Allocate output indices
    auto v_idx = torch::zeros({vertical_size}, torch::kInt32);
    auto s_idx = torch::zeros({slash_size}, torch::kInt32);

    // Call estimate function
    estimate_pattern_cpu(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        seq_len,
        head_dim,
        static_cast<int>(vertical_size),
        static_cast<int>(slash_size),
        v_idx.data_ptr<int>(),
        s_idx.data_ptr<int>()
    );

    return std::make_tuple(v_idx, s_idx);
}

// Compute-only function (given pre-computed pattern)
torch::Tensor sparse_attention_forward(
    torch::Tensor Q,           // [seq_len, head_dim]
    torch::Tensor K,           // [seq_len, head_dim]
    torch::Tensor V,           // [seq_len, head_dim]
    torch::Tensor v_idx,       // [num_vertical]
    torch::Tensor s_idx        // [num_slash]
) {
    TORCH_CHECK(Q.dim() == 2, "Q must be 2D tensor [seq_len, head_dim]");
    TORCH_CHECK(K.dim() == 2, "K must be 2D tensor");
    TORCH_CHECK(V.dim() == 2, "V must be 2D tensor");
    TORCH_CHECK(v_idx.dim() == 1, "v_idx must be 1D tensor");
    TORCH_CHECK(s_idx.dim() == 1, "s_idx must be 1D tensor");

    int seq_len = Q.size(0);
    int head_dim = Q.size(1);
    int num_vertical = v_idx.size(0);
    int num_slash = s_idx.size(0);

    auto output = torch::zeros_like(Q);

    sparse_attention_single_head(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        v_idx.data_ptr<int>(),
        s_idx.data_ptr<int>(),
        output.data_ptr<float>(),
        seq_len,
        head_dim,
        num_vertical,
        num_slash
    );

    return output;
}

// Performance analysis function
std::tuple<int, int, float, float> analyze_sparsity_pattern(
    int64_t seq_len,
    torch::Tensor v_idx,
    torch::Tensor s_idx
) {
    auto stats = analyze_sparsity(
        static_cast<int>(seq_len),
        v_idx.data_ptr<int>(),
        v_idx.size(0),
        s_idx.data_ptr<int>(),
        s_idx.size(0)
    );

    return std::make_tuple(
        stats.total_keys_computed,
        stats.total_keys_dense,
        stats.sparsity_ratio,
        stats.causal_savings
    );
}

}  // namespace cpu_sparse_attn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vertical_slash_attention",
          &cpu_sparse_attn::vertical_slash_attention_forward,
          "CPU Vertical-Slash Sparse Attention (estimate + compute)",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("vertical_size"), py::arg("slash_size"));

    m.def("estimate_pattern",
          &cpu_sparse_attn::estimate_pattern_forward,
          "Estimate sparse pattern from Q, K",
          py::arg("Q"), py::arg("K"),
          py::arg("vertical_size"), py::arg("slash_size"));

    m.def("sparse_attention",
          &cpu_sparse_attn::sparse_attention_forward,
          "Sparse attention with pre-computed pattern",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("v_idx"), py::arg("s_idx"));

    m.def("analyze_sparsity",
          &cpu_sparse_attn::analyze_sparsity_pattern,
          "Analyze sparsity pattern statistics",
          py::arg("seq_len"), py::arg("v_idx"), py::arg("s_idx"));
}
