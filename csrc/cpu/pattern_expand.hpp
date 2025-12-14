// Copyright (c) 2024-2025 Microsoft
// Licensed under The MIT License [see LICENSE for details]

#ifndef CPU_PATTERN_EXPAND_HPP
#define CPU_PATTERN_EXPAND_HPP

#include <vector>
#include <algorithm>

namespace cpu_sparse_attn {

// ============================================================================
// Pattern Expansion for Sparse Attention
// ============================================================================

// Represents a contiguous range of key positions [start, end)
struct KeyRange {
    int start;
    int end;  // exclusive

    KeyRange(int s, int e) : start(s), end(e) {}

    bool overlaps_or_adjacent(const KeyRange& other) const {
        return !(end < other.start || other.end < start);
    }

    void merge(const KeyRange& other) {
        start = std::min(start, other.start);
        end = std::max(end, other.end);
    }
};

// Merge a single key position into existing ranges
inline void merge_into_ranges(std::vector<KeyRange>& ranges, int key_pos) {
    KeyRange new_range(key_pos, key_pos + 1);

    // Find position to insert (keep sorted)
    auto it = std::lower_bound(ranges.begin(), ranges.end(), new_range,
        [](const KeyRange& a, const KeyRange& b) {
            return a.start < b.start;
        });

    // Check if we can merge with previous
    if (it != ranges.begin()) {
        auto prev = it - 1;
        if (prev->end >= key_pos) {
            // Extend previous range if needed
            prev->end = std::max(prev->end, key_pos + 1);
            // Check if we now overlap with next
            if (it != ranges.end() && prev->end >= it->start) {
                prev->end = std::max(prev->end, it->end);
                ranges.erase(it);
            }
            return;
        }
    }

    // Check if we can merge with next
    if (it != ranges.end() && it->start <= key_pos + 1) {
        it->start = std::min(it->start, key_pos);
        return;
    }

    // Insert new range
    ranges.insert(it, new_range);
}

// Expand pattern for a given query position
// Returns sorted, merged key ranges that need to be computed
inline std::vector<KeyRange> expand_pattern(
    int query_pos,
    const int* v_idx,      // [num_vertical] vertical column indices (sorted ascending)
    int num_vertical,
    const int* s_idx,      // [num_slash] diagonal offsets (sorted, typically descending)
    int num_slash
) {
    std::vector<KeyRange> ranges;
    ranges.reserve(num_vertical + num_slash);

    // Process Slash: diagonal offsets to key positions
    // s_idx[i] represents the i-th important diagonal
    // For query_pos q, the corresponding key position is: q - s_idx[i]
    for (int i = 0; i < num_slash; i++) {
        int key_pos = query_pos - s_idx[i];
        // Causal check: key_pos must be <= query_pos and >= 0
        if (key_pos >= 0 && key_pos <= query_pos) {
            merge_into_ranges(ranges, key_pos);
        }
    }

    // Process Vertical: direct column indices
    for (int i = 0; i < num_vertical; i++) {
        int key_pos = v_idx[i];
        // Causal check: key_pos must be <= query_pos
        if (key_pos <= query_pos) {
            merge_into_ranges(ranges, key_pos);
        }
    }

    return ranges;
}

// Count total valid keys for a query position (for performance analysis)
inline int count_valid_keys(
    int query_pos,
    const int* v_idx,
    int num_vertical,
    const int* s_idx,
    int num_slash
) {
    auto ranges = expand_pattern(query_pos, v_idx, num_vertical, s_idx, num_slash);
    int count = 0;
    for (const auto& r : ranges) {
        count += r.end - r.start;
    }
    return count;
}

// Pre-compute expanded patterns for all queries (optimization)
// This avoids repeated pattern expansion during attention computation
class PatternCache {
public:
    PatternCache(int seq_len, const int* v_idx, int num_vertical,
                 const int* s_idx, int num_slash)
        : seq_len_(seq_len), patterns_(seq_len) {

        #pragma omp parallel for
        for (int q = 0; q < seq_len; q++) {
            patterns_[q] = expand_pattern(q, v_idx, num_vertical, s_idx, num_slash);
        }
    }

    const std::vector<KeyRange>& get(int query_pos) const {
        return patterns_[query_pos];
    }

    // Get total number of (q,k) pairs to compute
    int total_pairs() const {
        int count = 0;
        for (int q = 0; q < seq_len_; q++) {
            for (const auto& r : patterns_[q]) {
                count += r.end - r.start;
            }
        }
        return count;
    }

    // Compare with dense causal attention
    float sparsity_ratio() const {
        int sparse_pairs = total_pairs();
        int dense_pairs = seq_len_ * (seq_len_ + 1) / 2;  // lower triangular
        return static_cast<float>(sparse_pairs) / dense_pairs;
    }

private:
    int seq_len_;
    std::vector<std::vector<KeyRange>> patterns_;
};

}  // namespace cpu_sparse_attn

#endif  // CPU_PATTERN_EXPAND_HPP
