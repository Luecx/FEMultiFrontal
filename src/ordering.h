/**
 * @file ordering.h
 * @brief Defines fill-reducing orderings for the independent multifrontal solver.
 *
 * The native ordering works on the symmetric sparsity pattern alone. It first
 * merges indistinguishable unknowns, i.e. unknowns with identical closed
 * adjacency, into weighted supervariables. Finite-element matrices usually
 * contain one such group per node, so the graph shrinks by the number of
 * surviving DOFs per node without any information supplied by the caller.
 *
 * The compressed graph is ordered by multilevel nested dissection: heavy-edge
 * coarsening, grown initial vertex separators, one-sided Fiduccia-Mattheyses
 * refinement during uncoarsening and halo minimum degree at the leaves.
 * Members of one supervariable remain consecutive in their original order.
 *
 * The ordering returns a scalar permutation; symbolic analysis and numerical
 * factorization consume that permutation without physical DOF block metadata.
 */
#pragma once

#include "matrix.h"

namespace mf {

// Return new-to-original indices, with subdomains before their separators.
// Indistinguishable unknowns are detected from the pattern and stay adjacent.
// Independent subdomains are ordered concurrently when threads exceeds one.
std::vector<int> nested_dissection(const Matrix& matrix, int threads = 1);



} // namespace mf
