/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// RISC-V RVV: forward all fast_scan specializations to NONE until
// dedicated RVV implementations are written.

#ifdef COMPILE_SIMD_RISCV_RVV

#include <faiss/impl/fast_scan/fast_scan.h>

namespace faiss {

template <>
void accumulate_to_mem_impl<SIMDLevel::RISCV_RVV>(
        int nq,
        size_t ntotal2,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        uint16_t* accu) {
    accumulate_to_mem_impl<SIMDLevel::NONE>(nq, ntotal2, nsq, codes, LUT, accu);
}

} // namespace faiss

#endif // COMPILE_SIMD_RISCV_RVV
