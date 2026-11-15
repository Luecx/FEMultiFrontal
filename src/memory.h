/**
 * @file memory.h
 * @brief Owns cache-line-aligned numerical storage and optional Linux page hints.
 *
 * Standard vector semantics are retained through a stateless allocator. Large
 * allocations may receive a best-effort Transparent Huge Page hint; ordinary
 * storage remains valid if the platform declines it. Successful advice is not
 * evidence that the kernel has allocated huge pages.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace mf::memory {

// Allocate numerical storage with 64-byte alignment, checking byte-count overflow
void* allocate(std::size_t count, std::size_t element_size);
void  release(void* pointer) noexcept;

// Advise only complete pages strictly inside the supplied allocation
bool advise_huge_pages(void* pointer, std::size_t bytes) noexcept;

// Process-wide resident anonymous huge pages in KiB; -1 means unavailable
std::int64_t anonymous_huge_pages_kib();

/**
 * @brief Stateless standard allocator for cache-line-aligned numerical vectors.
 *
 * All allocations use 64-byte alignment and are interchangeable across equal
 * allocator instances. This keeps vector move/swap ownership inexpensive for
 * frontal contributions. SIMD kernels must still respect row offsets: an
 * aligned base does not align every row in triangular packed storage.
 * Page-size policy is applied separately, without changing deallocation rules.
 *
 * @tparam T Numerical element type with alignment at most one cache line.
 */
template<class T>
struct AlignedAllocator {
    using value_type      = T;
    using is_always_equal = std::true_type;

    static_assert(alignof(T) <= 64);

    AlignedAllocator() noexcept = default;
    template<class U> AlignedAllocator(const AlignedAllocator<U>&) noexcept {}

    // Forward ownership to the common allocation implementation
    T* allocate(std::size_t count) { return static_cast<T*>(memory::allocate(count, sizeof(T))); }
    void deallocate(T* pointer, std::size_t) noexcept { memory::release(pointer); }

    template<class U> bool operator==(const AlignedAllocator<U>&) const noexcept { return true; }
    template<class U> bool operator!=(const AlignedAllocator<U>&) const noexcept { return false; }
};

using Values  = std::vector<double, AlignedAllocator<double>>;
using Indices = std::vector<int, AlignedAllocator<int>>;

/**
 * @brief Aligned storage for trivial scalars whose owner initializes every entry.
 *
 * Default construction starts the scalar lifetime without writing its value.
 * Explicit value construction and copying retain standard vector behavior.
 * This lets factor columns receive their first writes in the worker that owns
 * them, instead of serially touching the entire factor before task execution.
 * Every resized entry must be written before it is read; ordinary frontal
 * buffers continue using AlignedAllocator and normal value initialization.
 */
template<class T>
struct UninitializedAllocator : AlignedAllocator<T> {
    static_assert(std::is_trivial_v<T>);

    UninitializedAllocator() noexcept = default;
    template<class U> UninitializedAllocator(const UninitializedAllocator<U>&) noexcept {}

    // Preserve explicit values while avoiding implicit scalar zero initialization
    template<class U, class... Args>
    void construct(U* pointer, Args&&... args) {
        if constexpr (sizeof...(Args) == 0) {
            ::new (static_cast<void*>(pointer)) U;
        } else {
            ::new (static_cast<void*>(pointer)) U(std::forward<Args>(args)...);
        }
    }

    template<class U> bool operator==(const UninitializedAllocator<U>&) const noexcept { return true; }
    template<class U> bool operator!=(const UninitializedAllocator<U>&) const noexcept { return false; }
};

using UninitializedValues  = std::vector<double, UninitializedAllocator<double>>;
using UninitializedIndices = std::vector<int, UninitializedAllocator<int>>;

} // namespace mf::memory
