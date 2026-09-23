/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/scalar_quantizer/sq8_batch.h>

#include <algorithm>
#include <cmath>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/simd_dispatch.h>

namespace faiss {
namespace scalar_quantizer {

void sq8_batch_train(
        const ScalarQuantizer& sq,
        const float* query,
        bool l2,
        SQ8BatchWeights& w) {
    const bool uniform = sq.qtype == ScalarQuantizer::QT_8bit_uniform;
    FAISS_THROW_IF_NOT_MSG(
            uniform || sq.qtype == ScalarQuantizer::QT_8bit,
            "sq8_batch_train expects QT_8bit or QT_8bit_uniform");
    FAISS_THROW_IF_NOT(sq.trained.size() >= (uniform ? 2 : 2 * sq.d));

    // QT_8bit_uniform trains one [vmin, vdiff] for the whole vector, so the
    // per-dimension reads below collapse to a stride of zero.
    const float* vmin = sq.trained.data();
    const float* vdiff = sq.trained.data() + (uniform ? 1 : sq.d);
    const size_t stride = uniform ? 0 : 1;

    w.d = sq.d;
    w.l2 = l2;
    w.a.resize(sq.d);
    // A uniform range makes v_d a constant, carried in uniform_sq instead of a
    // vector the kernel would reload every sixteen lanes.
    w.b.resize(l2 && !uniform ? sq.d : 0);
    w.uniform_sq = 0;

    // Accumulated in double: the two halves of the L2 bias are large and of
    // opposite sign once the code term is added back, so a float accumulator
    // loses the difference between adjacent neighbours.
    double bias = 0;
    for (size_t i = 0; i < sq.d; i++) {
        const size_t j = i * stride;
        const float s = vdiff[j] / 255.f;
        if (l2) {
            const float a = query[i] - vmin[j] - 0.5f * s;
            bias += double(a) * a;
            w.a[i] = -2.f * a * s;
            if (uniform) {
                w.uniform_sq = s * s;
            } else {
                w.b[i] = s * s;
            }
        } else {
            bias += double(query[i]) * (vmin[j] + 0.5f * s);
            w.a[i] = query[i] * s;
        }
    }
    w.bias = float(bias);

    // A uniform range leaves one step `s` for every dimension, so the code's
    // contribution factors out of the sum and the dot product can be taken in
    // integers -- but only if the query is bytes too. Quantize it over its own
    // range and keep the float path's coefficients as well, so a caller that
    // cannot use the integer kernel is unaffected.
    w.uq.clear();
    if (!l2 && uniform && sq.d > 0) {
        float qmin = query[0];
        float qmax = query[0];
        for (size_t i = 1; i < sq.d; i++) {
            qmin = std::min(qmin, query[i]);
            qmax = std::max(qmax, query[i]);
        }
        const float qdiff = qmax - qmin;
        // A constant query has nothing to quantize; leave the integer path off
        // rather than divide by zero.
        if (qdiff > 0) {
            const float mid = 0.5f * (qmin + qmax);
            const float t = qdiff / 254.f;
            w.uq.resize(sq.d);
            for (size_t i = 0; i < sq.d; i++) {
                const long u = std::lround((query[i] - mid) / t);
                w.uq[i] = int8_t(std::clamp<long>(u, -127, 127));
            }
            const float s0 = vdiff[0] / 255.f;
            w.int_dot_scale = s0 * t;
            w.int_sum_scale = s0 * mid;
            w.int_bias = w.bias;
        }
    }
}

template <>
void sq8_batch_score4<SIMDLevel::NONE>(
        const SQ8BatchWeights& w,
        const uint8_t* const codes[4],
        float out[4],
        size_t d) {
    const float* a = w.a.data();
    const float* b = w.b.data();
    for (int k = 0; k < 4; k++) {
        const uint8_t* c = codes[k];
        float acc = 0;
        if (w.uniform_l2()) {
            uint32_t sq_sum = 0;
            for (size_t i = 0; i < d; i++) {
                acc += a[i] * float(c[i]);
                sq_sum += uint32_t(c[i]) * uint32_t(c[i]);
            }
            acc += w.uniform_sq * float(sq_sum);
        } else if (w.l2) {
            for (size_t i = 0; i < d; i++) {
                const float ci = float(c[i]);
                acc += a[i] * ci + b[i] * ci * ci;
            }
        } else {
            for (size_t i = 0; i < d; i++) {
                acc += a[i] * float(c[i]);
            }
        }
        const float r = w.bias + acc;
        out[k] = w.l2 ? -r : r;
    }
}

// A0 without RISCV_RVV: the kernel has AVX512, AVX2 and NEON
// specializations and falls back to the scalar loop elsewhere. AVX512_SPR
// resolves to AVX512 through get_simd_fallback.
constexpr int AVAILABLE_SIMD_LEVELS_SQ8_BATCH =
        AVAILABLE_SIMD_LEVELS_AVX2_NEON | (1 << int(SIMDLevel::AVX512));

void sq8_batch_score4_dispatch(
        const SQ8BatchWeights& w,
        const uint8_t* const codes[4],
        float out[4],
        size_t d) {
    with_selected_simd_levels<AVAILABLE_SIMD_LEVELS_SQ8_BATCH>(
            [&]<SIMDLevel SL>() { sq8_batch_score4<SL>(w, codes, out, d); });
}

} // namespace scalar_quantizer
} // namespace faiss
