#pragma once

#include <vector>
#include <cstdlib>
#include <new>

/**
 * A simple C++11-compatible aligned allocator using posix_memalign.
 * Essential for ensuring data lands on 64-byte boundaries for AVX-512 performance.
 */
template <typename T, std::size_t Alignment>
struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() noexcept = default;
    template <typename U> AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        void* ptr = nullptr;
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept { free(p); }

    template <typename U> struct rebind { using other = AlignedAllocator<U, Alignment>; };
    bool operator==(const AlignedAllocator&) const { return true; }
    bool operator!=(const AlignedAllocator&) const { return false; }
};

using TVector = std::vector<float>;
/** A float vector aligned to 64-byte boundaries for SIMD optimization. */
using TAlignedVector = std::vector<float, AlignedAllocator<float, 64>>;
