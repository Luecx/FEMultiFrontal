/**
 * @file dense_tiled.h
 * @brief Defines blocked lower-triangle storage for temporary Schur complements.
 *
 * Full 32-by-32 row-major tiles are stored by lower-triangular tile rows. Upper
 * entries of diagonal tiles and boundary padding are unused. This representation
 * changes temporary contributions only; scalar CSC factors and multifrontal
 * ownership are unchanged. The factorizer reserves physical index zero for the
 * already separated first pivot, aligning ordinary rank-32 updates to tiles.
 */
#pragma once

#include <cstddef>
#include <vector>

namespace mf::dense {

inline constexpr int schur_tile_size = 32;

// Return padded storage for a physical dimension, including the reserved pivot
inline std::size_t tiled_lower_size(int dimension) {
    const std::size_t tiles = (static_cast<std::size_t>(dimension) + schur_tile_size - 1) / schur_tile_size;
    return tiles * (tiles + 1) / 2 * schur_tile_size * schur_tile_size;
}

// Map physical indices row >= column into row-major lower-triangular tiles
inline std::size_t tiled_lower_offset(int row, int column) {
    const auto physical_row = static_cast<std::size_t>(row);
    const auto physical_col = static_cast<std::size_t>(column);
    const std::size_t block_row = physical_row / schur_tile_size;
    const std::size_t block_col = physical_col / schur_tile_size;
    return (block_row * (block_row + 1) / 2 + block_col) * schur_tile_size * schur_tile_size +
        (physical_row % schur_tile_size) * schur_tile_size + physical_col % schur_tile_size;
}

// Subtract X*X^T from the active suffix beginning at physical index origin+1.
// Scratch belongs to the calling numerical task and remains live through joins.
void rank_k_subtract_tiled(int n, int k, const double* const* columns, double* tiled,
                           int origin, std::vector<double>& scratch);

} // namespace mf::dense
