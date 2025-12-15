// Copyright (c) 2024-2025 Microsoft
// Licensed under The MIT License [see LICENSE for details]

#ifndef CPU_SPARSE_ATTENTION_HPP
#define CPU_SPARSE_ATTENTION_HPP

#include "vec_ops.hpp"
#include "pattern_expand.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#include <omp.h>

namespace cpu_sparse_attn {

// ============================================================================
// Estimate Stage: Extract sparse pattern from probe attention
// ============================================================================

// Diagonal sum using the same logic as sum_all_diagonal_matrix
// For probe attention [last_q, seq_len], compute sum along diagonals
inline void diagonal_sum(const float* qk, float* slash,
                         int last_q, int seq_len, int q_offset) {
    // Initialize slash to zero
    std::memset(slash, 0, seq_len * sizeof(float));

    // For each element in probe attention, add to corresponding diagonal
    // diagonal d = query_pos - key_pos
    for (int q = 0; q < last_q; q++) {
        int q_pos = q_offset + q;  // actual query position in full sequence
        for (int k = 0; k <= q_pos && k < seq_len; k++) {
            int diag = q_pos - k;
            if (diag < seq_len) {
                slash[diag] += qk[q * seq_len + k];
            }
        }
    }
}

// Estimate sparse pattern from Q, K
// Returns v_idx and s_idx arrays
void estimate_pattern_cpu(
    const float* Q,        // [seq_len, head_dim]
    const float* K,        // [seq_len, head_dim]
    int seq_len,
    int head_dim,
    int vertical_size,
    int slash_size,
    int* v_idx,            // output: [vertical_size]
    int* s_idx             // output: [slash_size]
) {
    const int last_q = std::min(64, seq_len);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int q_offset = seq_len - last_q;

    // Allocate probe attention matrix
    std::vector<float> qk(last_q * seq_len);

    // Step 1: Probe GEMM - Q[-last_q:] × K^T
    #pragma omp parallel for
    for (int q = 0; q < last_q; q++) {
        int q_pos = q_offset + q;
        for (int k = 0; k < seq_len; k++) {
            qk[q * seq_len + k] = gemv_avx512(
                Q + q_pos * head_dim,
                K + k * head_dim,
                head_dim
            ) * scale;
        }
    }

    // Step 2: Apply causal mask (set future positions to -inf)
    for (int q = 0; q < last_q; q++) {
        int q_pos = q_offset + q;
        for (int k = q_pos + 1; k < seq_len; k++) {
            qk[q * seq_len + k] = -std::numeric_limits<float>::infinity();
        }
    }

    // Step 3: Softmax per row
    for (int q = 0; q < last_q; q++) {
        softmax_avx512(qk.data() + q * seq_len, seq_len);
    }

    // Step 4: Vertical extraction - sum along query dimension
    std::vector<float> vertical(seq_len, 0.0f);
    sum_columns_avx512(qk.data(), vertical.data(), last_q, seq_len);

    // Force keep first 30 columns (BOS tokens, etc.)
    for (int k = 0; k < std::min(30, seq_len); k++) {
        vertical[k] = std::numeric_limits<float>::infinity();
    }

    // TopK for vertical
    int actual_v_size = std::min(vertical_size, seq_len);
    topk_indices(vertical.data(), seq_len, v_idx, actual_v_size);

    // Step 5: Slash extraction - diagonal sum
    std::vector<float> slash(seq_len, 0.0f);
    diagonal_sum(qk.data(), slash.data(), last_q, seq_len, q_offset);

    // Force keep last 100 diagonals (recent context)
    for (int d = std::max(0, seq_len - 100); d < seq_len; d++) {
        slash[d] = std::numeric_limits<float>::infinity();
    }

    // TopK for slash
    int actual_s_size = std::min(slash_size, seq_len);
    topk_indices(slash.data(), seq_len, s_idx, actual_s_size);
}

// ============================================================================
// Compute Stage: Sparse attention with fine-grained scheduling
// ============================================================================

// Single-head sparse attention kernel
void sparse_attention_single_head(
    const float* Q,        // [seq_len, head_dim]
    const float* K,        // [seq_len, head_dim]
    const float* V,        // [seq_len, head_dim]
    const int* v_idx,      // [num_vertical]
    const int* s_idx,      // [num_slash]
    float* output,         // [seq_len, head_dim]
    int seq_len,
    int head_dim,
    int num_vertical,
    int num_slash
) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Pre-compute pattern for all queries (optional optimization)
    // For now, compute on-the-fly

    #pragma omp parallel for schedule(dynamic, 64)
    for (int q = 0; q < seq_len; q++) {
        // Step 1: Expand pattern for this query
        auto key_ranges = expand_pattern(q, v_idx, num_vertical, s_idx, num_slash);

        // Step 2: Online softmax attention
        float m_i = -std::numeric_limits<float>::infinity();
        float l_i = 0.0f;
        std::vector<float> acc(head_dim, 0.0f);

        for (const auto& range : key_ranges) {
            for (int k = range.start; k < range.end; k++) {
                // GEMV: dot(Q[q], K[k])
                float score = gemv_avx512(Q + q * head_dim,
                                          K + k * head_dim,
                                          head_dim) * scale;

                // Online softmax update
                float m_new = std::max(m_i, score);
                float alpha = std::exp2f(m_i - m_new);
                float p = std::exp2f(score - m_new);

                // acc = acc * alpha + p * V[k]
                scale_and_add_avx512(acc.data(), alpha,
                                     V + k * head_dim, p, head_dim);

                l_i = l_i * alpha + p;
                m_i = m_new;
            }
        }

        // Step 3: Normalize output
        if (l_i > 0) {
            scale_avx512(output + q * head_dim, acc.data(), 1.0f / l_i, head_dim);
        } else {
            // No valid keys (shouldn't happen in practice)
            std::memset(output + q * head_dim, 0, head_dim * sizeof(float));
        }
    }
}

// Full vertical-slash attention kernel (estimate + compute)
void vertical_slash_attention_cpu(
    const float* Q,        // [batch, heads, seq_len, head_dim]
    const float* K,        // [batch, heads, seq_len, head_dim]
    const float* V,        // [batch, heads, seq_len, head_dim]
    float* output,         // [batch, heads, seq_len, head_dim]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int vertical_size,
    int slash_size
) {
    const int head_stride = seq_len * head_dim;
    const int batch_stride = num_heads * head_stride;

    // NOTE: No outer-level parallelism here to avoid nested parallelism overhead
    // Inner kernels (estimate_pattern_cpu and sparse_attention_single_head)
    // already parallelize over seq_len, which has much higher parallelism
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            const float* q_ptr = Q + b * batch_stride + h * head_stride;
            const float* k_ptr = K + b * batch_stride + h * head_stride;
            const float* v_ptr = V + b * batch_stride + h * head_stride;
            float* out_ptr = output + b * batch_stride + h * head_stride;

            // Allocate pattern indices
            std::vector<int> v_idx(vertical_size);
            std::vector<int> s_idx(slash_size);

            // Stage 1: Estimate pattern
            estimate_pattern_cpu(q_ptr, k_ptr, seq_len, head_dim,
                                 vertical_size, slash_size,
                                 v_idx.data(), s_idx.data());

            // Stage 2: Sparse attention
            sparse_attention_single_head(q_ptr, k_ptr, v_ptr,
                                         v_idx.data(), s_idx.data(),
                                         out_ptr, seq_len, head_dim,
                                         vertical_size, slash_size);
        }
    }
}

// ============================================================================
// Performance Analysis Utilities
// ============================================================================

struct PerformanceStats {
    int total_queries;
    int total_keys_computed;
    int total_keys_dense;
    float sparsity_ratio;
    float causal_savings;
};

PerformanceStats analyze_sparsity(
    int seq_len,
    const int* v_idx,
    int num_vertical,
    const int* s_idx,
    int num_slash
) {
    PerformanceStats stats;
    stats.total_queries = seq_len;
    stats.total_keys_computed = 0;

    for (int q = 0; q < seq_len; q++) {
        auto ranges = expand_pattern(q, v_idx, num_vertical, s_idx, num_slash);
        for (const auto& r : ranges) {
            stats.total_keys_computed += r.end - r.start;
        }
    }

    stats.total_keys_dense = seq_len * (seq_len + 1) / 2;  // causal
    stats.sparsity_ratio = static_cast<float>(stats.total_keys_computed)
                          / stats.total_keys_dense;

    // Estimate causal savings vs GPU block computation
    // GPU computes in 64x64 blocks, wastes ~50% on diagonal blocks
    int gpu_blocks = (seq_len + 63) / 64;
    int gpu_diagonal_waste = gpu_blocks * (64 * 64 - 64 * 65 / 2);
    stats.causal_savings = static_cast<float>(gpu_diagonal_waste)
                          / (gpu_blocks * 64 * 64);

    return stats;
}

}  // namespace cpu_sparse_attn

#endif  // CPU_SPARSE_ATTENTION_HPP
