/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/scalar_quantizer/sq8_batch.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/simd_dispatch.h>

namespace faiss {
namespace scalar_quantizer {

void sq8_batch_train(
        const ScalarQuantizer& sq,
        const float* query,
        bool l2,
        SQ8BatchWeights& w) {
    FAISS_THROW_IF_NOT_MSG(
            sq.qtype == ScalarQuantizer::QT_8bit,
            "sq8_batch_train expects a QT_8bit quantizer");
    FAISS_THROW_IF_NOT(sq.trained.size() >= 2 * sq.d);

    const float* vmin = sq.trained.data();
    const float* vdiff = sq.trained.data() + sq.d;

    w.d = sq.d;
    w.l2 = l2;
    w.a.resize(sq.d);
    w.b.resize(l2 ? sq.d : 0);

    // Accumulated in double: the two halves of the L2 bias are large and of
    // opposite sign once the code term is added back, so a float accumulator
    // loses the difference between adjacent neighbours.
    double bias = 0;
    for (size_t i = 0; i < sq.d; i++) {
        const float s = vdiff[i] / 255.f;
        if (l2) {
            const float a = query[i] - vmin[i] - 0.5f * s;
            bias += double(a) * a;
            w.a[i] = -2.f * a * s;
            w.b[i] = s * s;
        } else {
            bias += double(query[i]) * (vmin[i] + 0.5f * s);
            w.a[i] = query[i] * s;
        }
    }
    w.bias = float(bias);
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
        if (w.l2) {
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
