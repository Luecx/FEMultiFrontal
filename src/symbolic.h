/**
 * @file symbolic.h
 * @brief Defines scalar elimination fronts and the Cholesky fill pattern.
 *
 * Every matrix column owns exactly one pivot. Child update index sets are
 * assembled into their parent front, producing the elimination tree and exact
 * structural lower factor without performing numerical factorization.
 */
#pragma once

#include "matrix.h"
#include "memory.h"

namespace mf {

/**
 * @brief Immutable scalar-front assembly plan for one permuted sparsity pattern.
 *
 * A front consists of its column pivot followed by sorted update indices.
 * Factor rows include the diagonal first. Parent indices strictly increase;
 * therefore ascending columns are a valid child-before-parent traversal.
 * Separate child lists describe extend-add assembly, including disconnected
 * trees. reuse_child records exact index-set matches for buffer inheritance;
 * scalar fronts and factor columns remain explicit in this immutable plan.
 * Numerical Factor may group matching unary chains into multi-pivot panels
 * without changing this exact CSC pattern or requiring physical DOF blocks.
 */
struct Symbolic {
    // Compressed factor columns; analysis writes every row before publication
    std::vector<std::size_t>      ptr;
    memory::UninitializedIndices row;

    // Scalar elimination forest and numerical work estimates
    std::vector<int>         parent;
    std::vector<int>         first_child;
    std::vector<int>         next_child;
    std::vector<int>         reuse_child;
    std::size_t             max_front    = 0;
    double                  factor_flops = 0.0;

    // Child-before-parent traversal and contiguous subtree ranges. These retain
    // every scalar pivot; they only provide work units for a task scheduler.
    std::vector<int>    postorder;
    std::vector<int>    subtree_begin;
    std::vector<int>    subtree_size;
    std::vector<double> subtree_work;
    std::vector<int>    roots;

    // Construct exact fill using child unions, or parallel elimination-tree
    // reach when multiple workers are requested. Both return identical fronts.
    static Symbolic analyze(const Matrix& matrix, int threads = 1);
};

} // namespace mf
