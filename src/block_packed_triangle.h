#pragma once

#include "memory.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mf::dense {

// Lower block-triangular storage, with contiguous row-major dense tiles.
// Diagonal tiles are stored as full squares to make tile addressing cheap.
// Row/column indices are LOCAL to the owning contribution allocation. An
// inherited front retains its buffer; the caller shifts its origin instead.
class BlockPackedTriangle {
public:
    static constexpr int tile_size = 48;

    void assign_triangle(int n) {
        if (n < 0) { throw std::invalid_argument("negative BPT dimension"); }
        n_ = n;
        blocks_ = (n + tile_size - 1) / tile_size;
        offsets_.resize(static_cast<std::size_t>(blocks_) * (blocks_ + 1) / 2);
        std::size_t length = 0;
        for (int bi = 0; bi < blocks_; ++bi) {
            const int rows = block_length(bi);
            for (int bj = 0; bj <= bi; ++bj) {
                offsets_[block_id(bi, bj)] = length;
                length += static_cast<std::size_t>(rows) * block_length(bj);
            }
        }
        values_.assign(length, 0.0);
    }

    int dimension() const noexcept { return n_; }
    std::size_t size() const noexcept { return values_.size(); }
    double* data() noexcept { return values_.data(); }
    const double* data() const noexcept { return values_.data(); }

    double& at(int i, int j) noexcept {
        const int bi = i / tile_size, bj = j / tile_size;
        const int col_length = block_length(bj);
        return values_[offsets_[block_id(bi, bj)]
                     + static_cast<std::size_t>(i - bi * tile_size) * col_length
                     + (j - bj * tile_size)];
    }
    const double& at(int i, int j) const noexcept {
        const int bi = i / tile_size, bj = j / tile_size;
        const int col_length = block_length(bj);
        return values_[offsets_[block_id(bi, bj)]
                     + static_cast<std::size_t>(i - bi * tile_size) * col_length
                     + (j - bj * tile_size)];
    }

    // Block-wise access. Each tile is row-major, with a stride equal to the
    // actual (not necessarily full) width of its column block.
    double* block_data(int bi, int bj) noexcept { return data() + offsets_[block_id(bi, bj)]; }
    int block_length(int b) const noexcept { return std::min(tile_size, n_ - b * tile_size); }
    int block_count() const noexcept { return blocks_; }

    void clear() noexcept {
        memory::Values().swap(values_);
        std::vector<std::size_t>().swap(offsets_);
        n_ = blocks_ = 0;
    }

private:
    static std::size_t block_id(int i, int j) noexcept {
        return static_cast<std::size_t>(i) * (i + 1) / 2 + j;
    }
    int n_ = 0;
    int blocks_ = 0;
    std::vector<std::size_t> offsets_;
    memory::Values values_;
};

// Assemble the already-eliminated contribution into the group's remaining
// CSC columns. The contribution has a leading origin offset from inheritance.
void add_block_packed_panel(int size, int width, double* const* columns,
                            const BlockPackedTriangle& contribution, int origin);

// In-place Schur update in the lower block triangle. All tile writes are
// disjoint, so OpenMP tasking over block rows is safe in principle.
void rank_k_subtract_bpt(int n, int k, const double* const* columns,
                         BlockPackedTriangle& contribution, int origin);

} // namespace mf::dense
