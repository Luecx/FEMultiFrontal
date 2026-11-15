/**
 * @file pivot_driver.h
 * @brief Declares the explicit robust-mode benchmark entry point.
 *
 * The existing SPD driver remains the default. This separate path imports
 * canonical symmetric data for LDLT or full unmodified general data for LU,
 * then reports threshold rank and independent residual checks.
 */
#pragma once

namespace mf {

// Execute explicit symmetric/PSD/general modes using the same standalone binary
int run_pivot_driver(int argc, char** argv);

} // namespace mf
