/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/hnsw/LockVector.h>

#include <algorithm>

namespace faiss {

LockVector::LockVector(LockVector&& other) noexcept
        : size_(other.size_), capacity_(other.capacity_) {
    other.size_ = 0;
    other.capacity_ = 0;
}

void LockVector::prepare(size_t new_size) {
    if (new_size <= size_) {
        return;
    }
    if (new_size > capacity_) {
        capacity_ = std::max(new_size, capacity_ * 2);
    }
    size_ = new_size;
}

void LockVector::clear() {
    size_ = 0;
    capacity_ = 0;
}

} // namespace faiss
