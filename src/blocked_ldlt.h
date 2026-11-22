/**
 * @file blocked_ldlt.h
 * @brief Blocked, threshold-pivoted multifrontal LDLT with delayed variables.
 */
#pragma once

namespace mf {
struct PivotFactor;
struct GeneralMatrix;
struct Symbolic;

void factorize_blocked_ldlt(PivotFactor& factor, const GeneralMatrix& input,
                            const Symbolic& symbolic);
} // namespace mf
