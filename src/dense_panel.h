/**
 * @file dense_panel.h
 * @brief Dense panel assembly and POTRF/TRSM for multifrontal SPD fronts.
 *
 * Scalar CSC columns persist as the factor representation. These helpers read
 * the packed contribution panel once, factor its leading dense block and solve
 * the rectangular suffix. Front ownership and Schur updates remain the caller's
 * responsibility. No BLAS, physical block metadata or supernodal storage is used.
 */
#pragma once

namespace mf::dense {

// Add packed contribution columns 1..width-1 to initialized CSC front columns.
// Column zero has already been assembled. origin locates the subtriangle that
// begins at local front index one; each column j contains size-j valid values.
void add_packed_panel(int size, int width, double* const* columns, const double* packed, int origin);

// Factor the leading width-by-width SPD block. Width lies in [1,256]. inverse
// receives width reciprocal diagonal values. Return -1 on success, otherwise
// the local nonpositive/nonfinite pivot index; minimum tracks tested pivots.
int factor_pivot_block(int width, double* const* columns, double* inverse, double& minimum);

// Solve F21*inv(L11^T) for rows width..size-1, using the completed leading block
// and its reciprocal diagonals. Columns retain their scalar CSC representation.
void solve_panel(int size, int width, double* const* columns, const double* inverse);

// Forward substitution on up to eight consecutive columns with one common row
// list. Column j stores rows[j..size); rows are sorted and distinct. If rhs is
// non-null, add rhs[rows[j]] immediately before solving pivot j in private work;
// otherwise work already contains the RHS. Factor columns remain read-only.
void forward_panel(int size, int width, const int* rows, const double* const* columns,
                   double* work, const double* rhs);

} // namespace mf::dense
