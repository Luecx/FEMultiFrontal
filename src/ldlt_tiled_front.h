/**
 * @file ldlt_tiled_front.h
 * @brief Optional 32x32 tile-major storage for symmetric multifrontal fronts.
 *
 * All off-diagonal tiles store dense row-major 32x32 doubles. Diagonal tiles
 * also occupy 32x32 doubles, with only lower-triangular entries meaningful.
 * Physical offsets are preserved across inherited parent-front tails.
 * Small fronts keep the existing row-packed triangle to avoid tile padding.
 */
#pragma once

#include <cstddef>
#include <vector>

namespace mf::dense {

inline constexpr int front_tile_size = 32;
inline constexpr std::size_t front_tile_entries = 32u * 32u;

inline std::size_t tiled_front_entries(std::size_t physical_width) noexcept {
    const std::size_t tiles = (physical_width + front_tile_size - 1) / front_tile_size;
    return (tiles * (tiles + 1) / 2) * front_tile_entries;
}

// Requires physical_row >= physical_col, both nonnegative.
inline std::size_t tiled_front_offset(int physical_row, int physical_col) noexcept {
    const std::size_t tr = static_cast<std::size_t>(physical_row) / front_tile_size;
    const std::size_t tc = static_cast<std::size_t>(physical_col) / front_tile_size;
    const std::size_t tile = tr * (tr + 1) / 2 + tc;
    return tile * front_tile_entries +
        static_cast<std::size_t>(physical_row % front_tile_size) * front_tile_size +
        static_cast<std::size_t>(physical_col % front_tile_size);
}

// C(active,active) -= left(active,:) * right(active,:)^T.
// C has 32x32 tile-major lower-triangle layout, rather than ordinary packed
// row-triangle layout. origin shifts local active indices into physical storage.
// Arbitrary sorted active indices, delayed-pivot holes, and signed/2x2 D
// panels are supported; no inactive/upper-triangle entries are modified.
void rank_k_ldlt_tiled_front(int origin, int dim, const std::vector<int>& active,
                             int rank, const double* left, const double* right,
                             double* tile_packed);

} // namespace mf::dense
