/**
 * @file cache_matrix.cpp
 * @brief Saves a validated Matrix Market input in warp9's native CSC format.
 *
 * Text parsing, symmetry validation and duplicate summation use the production
 * importer. The resulting cache is host-specific and contains no solver state.
 */
#include "load_matrix.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr << "Usage: cache_matrix input.mtx output.csc\n";
            return 2;
        }
        const auto matrix = load_benchmark_matrix(argv[1]);
        std::ofstream file(argv[2], std::ios::binary);
        const std::size_t entries = matrix.row.size();
        file.write(reinterpret_cast<const char*>(&matrix.n), sizeof(matrix.n));
        file.write(reinterpret_cast<const char*>(&entries), sizeof(entries));
        file.write(reinterpret_cast<const char*>(matrix.ptr.data()), matrix.ptr.size() * sizeof(std::size_t));
        file.write(reinterpret_cast<const char*>(matrix.row.data()), entries * sizeof(int));
        file.write(reinterpret_cast<const char*>(matrix.value.data()), entries * sizeof(double));
        file.close();
        if (!file) { throw std::runtime_error("Failed to write matrix cache"); }
        std::cout << "n=" << matrix.n << " lower_nnz=" << entries << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
