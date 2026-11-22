/**
 * @file metis_ordering.h
 * @brief Optional METIS_NodeND baseline for fill-reducing order comparisons.
 */
#pragma once

#include "matrix.h"

namespace mf {

// Return order[new] = old, matching nested_dissection() and Matrix::permute().
// This entry point throws a useful diagnostic when compiled without METIS.
std::vector<int> metis_nested_dissection(const Matrix& matrix);

} // namespace mf
