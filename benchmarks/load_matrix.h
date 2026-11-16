/**
 * @file load_matrix.h
 * @brief Imports Matrix Market inputs or the native CSC caches used on warp9.
 *
 * Cache files store int n, size_t nnz, then native ptr/row/value arrays. They
 * are specific to the host byte order and index sizes. Matrix validation and
 * all import work remain outside numerical benchmark timings.
 */
#pragma once

#include "matrix.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

inline mf::Matrix load_benchmark_matrix(const std::string& path) {
    if (std::filesystem::path(path).extension() != ".csc") {
        return mf::Matrix::read_market(path);
    }
    std::ifstream file(path, std::ios::binary);
    mf::Matrix matrix;
    std::size_t entries = 0;
    file.read(reinterpret_cast<char*>(&matrix.n), sizeof(matrix.n));
    file.read(reinterpret_cast<char*>(&entries), sizeof(entries));
    const auto bytes = std::filesystem::file_size(path);
    if (!file || matrix.n <= 0 || entries > bytes / 12 ||
        static_cast<std::size_t>(matrix.n) > bytes / sizeof(std::size_t)) {
        throw std::runtime_error("Invalid CSC cache header");
    }
    matrix.ptr.resize(static_cast<std::size_t>(matrix.n) + 1);
    matrix.row.resize(entries);
    matrix.value.resize(entries);
    file.read(reinterpret_cast<char*>(matrix.ptr.data()), matrix.ptr.size() * sizeof(std::size_t));
    file.read(reinterpret_cast<char*>(matrix.row.data()), entries * sizeof(int));
    file.read(reinterpret_cast<char*>(matrix.value.data()), entries * sizeof(double));
    if (!file || file.peek() != std::char_traits<char>::eof() || matrix.ptr[0] != 0 || matrix.ptr.back() != entries) {
        throw std::runtime_error("Truncated or inconsistent CSC cache");
    }
    for (int c = 0; c < matrix.n; ++c) {
        if (matrix.ptr[c] > matrix.ptr[c+1] || matrix.ptr[c+1] > entries) {
            throw std::runtime_error("Invalid CSC column offsets");
        }
        int previous = c - 1;
        for (std::size_t p = matrix.ptr[c]; p < matrix.ptr[c+1]; ++p) {
            if (matrix.row[p] <= previous || matrix.row[p] >= matrix.n || !std::isfinite(matrix.value[p])) {
                throw std::runtime_error("Invalid CSC row or value");
            }
            previous = matrix.row[p];
        }
    }
    return matrix;
}
