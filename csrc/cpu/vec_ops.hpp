// Copyright (c) 2024-2025 Microsoft
// Licensed under The MIT License [see LICENSE for details]

#ifndef CPU_VEC_OPS_HPP
#define CPU_VEC_OPS_HPP

#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <limits>

namespace cpu_sparse_attn {

// ============================================================================
// AVX-512 Vector Operations for Sparse Attention
// ============================================================================

// 128-dim dot product using AVX-512 (8 FMA operations)
inline float gemv_avx512(const float* a, const float* b, int dim) {
    __m512 acc = _mm512_setzero_ps();

    // 128 / 16 = 8 iterations for head_dim=128
    for (int i = 0; i < dim; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        acc = _mm512_fmadd_ps(va, vb, acc);
    }

    return _mm512_reduce_add_ps(acc);
}

// Scale vector: out = in * scale
inline void scale_avx512(float* out, const float* in, float scale, int dim) {
    __m512 vs = _mm512_set1_ps(scale);

    for (int i = 0; i < dim; i += 16) {
        __m512 vin = _mm512_loadu_ps(in + i);
        __m512 vout = _mm512_mul_ps(vin, vs);
        _mm512_storeu_ps(out + i, vout);
    }
}

// acc = acc * alpha + v * p
inline void scale_and_add_avx512(float* acc, float alpha,
                                  const float* v, float p, int dim) {
    __m512 va = _mm512_set1_ps(alpha);
    __m512 vp = _mm512_set1_ps(p);

    for (int i = 0; i < dim; i += 16) {
        __m512 vacc = _mm512_loadu_ps(acc + i);
        __m512 vv = _mm512_loadu_ps(v + i);
        // acc = acc * alpha + v * p
        vacc = _mm512_fmadd_ps(vacc, va, _mm512_mul_ps(vv, vp));
        _mm512_storeu_ps(acc + i, vacc);
    }
}

// Vectorized softmax for a single row
inline void softmax_avx512(float* data, int len) {
    // Step 1: Find max
    float max_val = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < len; i++) {
        max_val = std::max(max_val, data[i]);
    }

    // Step 2: exp(x - max) and sum
    __m512 vmax = _mm512_set1_ps(max_val);
    __m512 vsum = _mm512_setzero_ps();

    int i = 0;
    for (; i + 16 <= len; i += 16) {
        __m512 vx = _mm512_loadu_ps(data + i);
        vx = _mm512_sub_ps(vx, vmax);
        // Use exp approximation or call expf
        // For simplicity, use scalar exp here
        float tmp[16];
        _mm512_storeu_ps(tmp, vx);
        for (int j = 0; j < 16; j++) {
            tmp[j] = expf(tmp[j]);
        }
        vx = _mm512_loadu_ps(tmp);
        _mm512_storeu_ps(data + i, vx);
        vsum = _mm512_add_ps(vsum, vx);
    }
    // Handle remainder
    float sum = _mm512_reduce_add_ps(vsum);
    for (; i < len; i++) {
        data[i] = expf(data[i] - max_val);
        sum += data[i];
    }

    // Step 3: Normalize
    float inv_sum = 1.0f / sum;
    __m512 vinv = _mm512_set1_ps(inv_sum);
    i = 0;
    for (; i + 16 <= len; i += 16) {
        __m512 vx = _mm512_loadu_ps(data + i);
        vx = _mm512_mul_ps(vx, vinv);
        _mm512_storeu_ps(data + i, vx);
    }
    for (; i < len; i++) {
        data[i] *= inv_sum;
    }
}

// Find top-k indices (simple implementation)
inline void topk_indices(const float* data, int len, int* indices, int k) {
    // Create index array
    std::vector<std::pair<float, int>> indexed(len);
    for (int i = 0; i < len; i++) {
        indexed[i] = {data[i], i};
    }

    // Partial sort to get top-k
    std::partial_sort(indexed.begin(), indexed.begin() + k, indexed.end(),
                      [](const auto& a, const auto& b) {
                          return a.first > b.first;
                      });

    // Extract indices
    for (int i = 0; i < k; i++) {
        indices[i] = indexed[i].second;
    }
}

// Sum along columns (for vertical extraction)
inline void sum_columns_avx512(const float* mat, float* out,
                                int rows, int cols) {
    // Initialize output to zero
    for (int j = 0; j < cols; j++) {
        out[j] = 0.0f;
    }

    // Accumulate each row
    for (int i = 0; i < rows; i++) {
        int j = 0;
        for (; j + 16 <= cols; j += 16) {
            __m512 vout = _mm512_loadu_ps(out + j);
            __m512 vrow = _mm512_loadu_ps(mat + i * cols + j);
            vout = _mm512_add_ps(vout, vrow);
            _mm512_storeu_ps(out + j, vout);
        }
        for (; j < cols; j++) {
            out[j] += mat[i * cols + j];
        }
    }
}

}  // namespace cpu_sparse_attn

#endif  // CPU_VEC_OPS_HPP
