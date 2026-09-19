/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/**
 * @file sq8_batch.h
 * @brief Batched query-to-code scoring for QT_8bit scalar quantization.
 *
 * A QT_8bit code decodes as x_d = vmin_d + (c_d + 0.5) * vdiff_d / 255, one
 * range per dimension, so both similarities are polynomials in the raw code
 * bytes whose coefficients depend only on the query:
 *
 *   <q, x>      = C + sum_d w_d c_d,          w_d = q_d * s_d
 *   ||q - x||^2 = A + sum_d u_d c_d + sum_d v_d c_d^2
 *
 * with s_d = vdiff_d / 255, a_d = q_d - vmin_d - s_d/2, u_d = -2 a_d s_d,
 * v_d = s_d^2, C = sum_d q_d (vmin_d + s_d/2) and A = sum_d a_d^2.
 *
 * Folding the decode into the query this way means the codes are never
 * materialised as floats: the bytes widen straight into the FMA. Four codes
 * are scored side by side so the dependent accumulator chains overlap, which
 * is what the batch is for -- one code at a time leaves the FMA units idle
 * waiting on the chain.
 *
 * This is the scoring shape a graph or cluster scan wants, where the codes to
 * score arrive as a list of pointers rather than as a contiguous block, so
 * neither the InvertedListScanner nor a DistanceComputer fits.
 */

#include <cstddef>
#include <cstdint>
#include <vector>

#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/utils/simd_levels.h>

namespace faiss {
namespace scalar_quantizer {

/// Per-query coefficients for QT_8bit scoring. Built once per query by
/// sq8_batch_train, then reused for every code scored against it.
struct SQ8BatchWeights {
    /// w_d (inner product) or u_d (L2).
    std::vector<float> a;
    /// v_d; empty for inner product and for the uniform quantizer, whose v_d
    /// is the single constant in uniform_sq.
    std::vector<float> b;
    /// C (inner product) or A (L2).
    float bias = 0;
    /// Set when every dimension shares one range (QT_8bit_uniform): then
    /// v_d = s^2 is the same constant for all d, so sum_d v_d c_d^2 becomes
    /// uniform_sq * sum_d c_d^2 and the squared term accumulates in integers
    /// rather than as a second float FMA chain. Zero for inner product and
    /// for the per-dimension quantizer.
    float uniform_sq = 0;
    /// Number of dimensions the coefficients cover.
    size_t d = 0;
    /// True when the coefficients encode squared L2 rather than inner product.
    bool l2 = false;

    /// Inner product against a uniform quantizer, with the query quantized to
    /// bytes as well, so the dot product runs in integers: 64 dimensions to an
    /// AVX-512 VNNI instruction against 16 for the float chain the per-
    /// dimension coefficients need. Empty when the path does not apply -- a
    /// per-dimension range has no single s to factor out, and L2 keeps its own
    /// arrangement above.
    ///
    /// The reconstruction, with s the code step, t the query step and
    /// D = sum u_d c_d, Sc = sum c_d, Sq = sum qhat_d:
    ///
    ///   ip = vmin*Sq + 0.5*s*Sq + s*(qmin*Sc + t*D + 0.5*t*Sc)
    ///
    /// which is `int_dot * int_dot_scale + Sc * int_sum_scale + int_bias`.
    std::vector<uint8_t> uq;
    float int_dot_scale = 0;
    float int_sum_scale = 0;
    float int_bias = 0;

    /// True when `uq` holds a quantized query and the integer kernel applies.
    bool uniform_ip() const {
        return !l2 && !uq.empty();
    }

    /// True when the L2 squared term is the single constant uniform_sq rather
    /// than the per-dimension vector b, which is then empty. The distinction
    /// cannot be read off uniform_sq's value: a corpus whose vectors are all
    /// identical trains vdiff = 0, so the constant is legitimately zero while
    /// b is still absent.
    bool uniform_l2() const {
        return l2 && b.empty();
    }
};

/** Build the per-query coefficients.
 *
 * Accepts QT_8bit (per-dimension [vmin, vdiff]) and QT_8bit_uniform (one
 * shared [vmin, vdiff]); the uniform case additionally sets uniform_sq.
 *
 * @param sq     a trained QT_8bit or QT_8bit_uniform quantizer
 * @param query  sq.d floats
 * @param l2     squared L2 when true, inner product when false
 * @param w      output, resized as needed
 */
void sq8_batch_train(
        const ScalarQuantizer& sq,
        const float* query,
        bool l2,
        SQ8BatchWeights& w);

/** Score four codes against the coefficients.
 *
 * Returns a similarity in both cases: the inner product itself, or the
 * negated squared L2, so larger is always closer.
 *
 * @param w      coefficients from sq8_batch_train
 * @param codes  four code pointers; entries may repeat to pad a short tail
 * @param out    four similarities
 * @param d      dimensions to score, <= w.d (a prefix scores a prefix)
 */
template <SIMDLevel SL0>
void sq8_batch_score4(
        const SQ8BatchWeights& w,
        const uint8_t* const codes[4],
        float out[4],
        size_t d);

/// Runtime-dispatched sq8_batch_score4.
void sq8_batch_score4_dispatch(
        const SQ8BatchWeights& w,
        const uint8_t* const codes[4],
        float out[4],
        size_t d);

/** Score `n` codes, four at a time, prefetching ahead.
 *
 * `get_code(i)` returns the i-th code. The codes a scan visits are spread
 * over the code array, so without the prefetch every group of four waits on
 * memory; only the bytes the kernel reads are touched.
 */
template <typename GetCode>
void sq8_batch_score_n(
        const SQ8BatchWeights& w,
        GetCode&& get_code,
        size_t n,
        float* out,
        size_t d) {
    if (d == 0 || d > w.d) {
        d = w.d;
    }
    constexpr size_t lookahead = 8;
    // Touch the head of the code and let the hardware stream prefetcher carry
    // the rest: a code is read front to back, so one miss is all the prefetcher
    // needs to lock on. Asking for every line of it instead fills the load
    // queue with requests that are dropped long before they are used -- at
    // d=1024 that was sixteen prefetches a candidate, eight candidates deep,
    // and it cost more than the distance kernel it was feeding.
    constexpr size_t kPrefetchBytes = 128;
    const size_t prefetch_bytes = d < kPrefetchBytes ? d : kPrefetchBytes;
    auto prefetch = [&](size_t j) {
        if (j < n) {
            const uint8_t* p = get_code(j);
            for (size_t off = 0; off < prefetch_bytes; off += 64) {
                __builtin_prefetch(p + off, 0, 1);
            }
        }
    };
    for (size_t j = 0; j < lookahead && j < n; j++) {
        prefetch(j);
    }
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        for (size_t j = i + lookahead; j < i + lookahead + 4; j++) {
            prefetch(j);
        }
        const uint8_t* codes[4] = {
                get_code(i), get_code(i + 1), get_code(i + 2), get_code(i + 3)};
        sq8_batch_score4_dispatch(w, codes, out + i, d);
    }
    if (i < n) {
        const uint8_t* codes[4] = {
                get_code(i), get_code(i), get_code(i), get_code(i)};
        for (size_t k = 1; i + k < n; k++) {
            codes[k] = get_code(i + k);
        }
        float tail[4];
        sq8_batch_score4_dispatch(w, codes, tail, d);
        for (size_t k = 0; i + k < n; k++) {
            out[i + k] = tail[k];
        }
    }
}

} // namespace scalar_quantizer
} // namespace faiss
