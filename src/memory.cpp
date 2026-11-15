/**
 * @file memory.cpp
 * @brief Implements aligned allocations and observable best-effort page advice.
 *
 * Numerical buffers use normal free-compatible POSIX allocations, not a
 * reserved hugetlb pool. Linux MADV_HUGEPAGE is optional and limited to complete
 * owned pages. A process-level smaps summary distinguishes accepted hints from
 * actual resident anonymous huge pages without claiming per-buffer attribution.
 */
#include "memory.h"

#include <cstdlib>
#include <fstream>
#include <limits>
#include <new>
#include <string>

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace mf::memory {

/**
 * Allocate count numerical entries on a 64-byte boundary.
 *
 * Reject byte-count overflow before multiplication. Zero storage has no owned
 * address; allocation failure follows the standard allocator bad_alloc contract.
 */
void* allocate(std::size_t count, std::size_t element_size) {
    // Validate the multiplication before asking the allocator for storage
    if (element_size == 0 || count > std::numeric_limits<std::size_t>::max() / element_size) {
        throw std::bad_array_new_length();
    }
    if (count == 0) { return nullptr; }

    // Keep normal free-compatible ownership while aligning the numerical base
    void* pointer = nullptr;
    if (posix_memalign(&pointer, 64, count * element_size) != 0) { throw std::bad_alloc(); }
    return pointer;
}

void release(void* pointer) noexcept {
    std::free(pointer);
}

/**
 * Request Transparent Huge Pages for complete pages inside a large allocation.
 *
 * The first and last partial pages are excluded to avoid advising allocator
 * metadata or neighboring objects. A successful return reports accepted advice,
 * not actual huge-page residency. Unsupported platforms retain normal storage.
 */
bool advise_huge_pages(void* pointer, std::size_t bytes) noexcept {
#ifdef __linux__
    // Ignore small buffers and obtain the actual base page size
    if (!pointer || bytes < 2 * 1024 * 1024) { return false; }
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) { return false; }

    // Round inward so every advised byte belongs to this numerical allocation
    const auto page  = static_cast<std::uintptr_t>(page_size);
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    const auto first = (begin + page - 1) / page * page;
    const auto last  = (begin + bytes) / page * page;
    if (last <= first || last - first < 2 * 1024 * 1024) { return false; }
    return madvise(reinterpret_cast<void*>(first), last - first, MADV_HUGEPAGE) == 0;
#else
    (void)pointer;
    (void)bytes;
    return false;
#endif
}

/**
 * Inspect process-wide resident anonymous huge pages when Linux exposes smaps.
 *
 * The count includes all mappings in this process and cannot be attributed to
 * a particular factor or thread. Missing information is reported as unavailable.
 */
std::int64_t anonymous_huge_pages_kib() {
    // Inspect only this process, without requiring system configuration changes
    std::ifstream input("/proc/self/smaps_rollup");
    std::string key;
    std::string rest;
    while (input >> key) {
        if (key == "AnonHugePages:") {
            std::int64_t value = 0;
            if (input >> value) { return value; }
            return -1;
        }
        std::getline(input, rest);
    }
    return -1;
}

} // namespace mf::memory
