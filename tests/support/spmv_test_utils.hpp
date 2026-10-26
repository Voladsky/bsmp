#ifndef BSMP_TESTS_SPMV_TEST_UTILS_HPP
#define BSMP_TESTS_SPMV_TEST_UTILS_HPP

// Shared test support for the bsmp HPC framework test suite.
//
// Everything lives in the common `spmv` namespace: the framework will grow
// more functionality (solvers, formats, workspace clients), and every future
// test file should reuse the helpers defined here instead of re-implementing
// references or RNGs.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "core/constants.hpp"
#include "core/dense_vector.hpp"
#include "io/parsers_utilities.hpp"

namespace spmv {

using Vec = std::vector<float>;

// ---------------------------------------------------------------------------
// Deterministic RNG (MMIX LCG). Fixed seeds -> bit-identical matrices on every
// platform and every run, so test failures are reproducible.
// ---------------------------------------------------------------------------
class Lcg {
   public:
    explicit Lcg(uint64_t seed) : state_(seed != 0 ? seed : 0x9e3779b97f4a7c15ull) {}

    uint32_t next() {
        state_ = state_ * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint32_t>(state_ >> 33);
    }

    // Uniform in [0, 1) with 24-bit mantissa precision.
    float next_float01() {
        constexpr float kScale = 1.0f / 16777216.0f;  // 2^24
        return static_cast<float>(next() & 0x00FFFFFFu) * kScale;
    }

    // Uniform in [lo, hi).
    int next_int(int lo, int hi) {
        return lo + static_cast<int>(next() % static_cast<uint32_t>(hi - lo));
    }

   private:
    uint64_t state_;
};

// ---------------------------------------------------------------------------
// Host reference for block-COO SpMV (element space, mirrors the kernel
// semantics: partial edge blocks are masked by num_rows / num_cols).
// ---------------------------------------------------------------------------
inline void host_spmv(const bsmp::io::parser_utilities::HostCOO& coo,
                      const Vec& x, Vec& y) {
    const int block = coo.block_size;
    y.assign(static_cast<size_t>(coo.num_rows), 0.0f);
    for (size_t b = 0; b < coo.block_rows.size(); ++b) {
        const int brow = coo.block_rows[b];
        const int bcol = coo.block_cols[b];
        const float* data = coo.block_data.data() + b * block * block;
        for (int r = 0; r < block; ++r) {
            const int row = brow * block + r;
            if (row >= coo.num_rows) continue;
            float acc = 0.0f;
            for (int c = 0; c < block; ++c) {
                const int col = bcol * block + c;
                if (col >= coo.num_cols) continue;
                acc += data[r * block + c] * x[static_cast<size_t>(col)];
            }
            y[static_cast<size_t>(row)] += acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Matrix builders (all produce HostCOO, i.e. the parser output shape).
// ---------------------------------------------------------------------------

// Diagonal matrix of ones: one dense identity block per block row.
inline bsmp::io::parser_utilities::HostCOO make_identity_coo(int n) {
    using bsmp::io::parser_utilities::HostCOO;
    HostCOO coo{};
    const int b = bsmp::core::constants::BLOCK_SIZE;
    coo.block_size = b;
    coo.num_rows = n;
    coo.num_cols = n;
    const int num_block_rows = (n + b - 1) / b;
    for (int brow = 0; brow < num_block_rows; ++brow) {
        const size_t block_begin = coo.block_data.size();
        coo.block_rows.push_back(brow);
        coo.block_cols.push_back(brow);
        coo.block_data.insert(coo.block_data.end(),
                              static_cast<size_t>(b) * b, 0.0f);
        for (int k = 0; k < b; ++k) {
            coo.block_data[block_begin + static_cast<size_t>(k) * b + k] = 1.0f;
        }
    }
    return coo;
}

// Random element-level triplets routed through the production parser, so the
// grouping/builder logic is exactly what parse_coo does.
inline bsmp::io::parser_utilities::HostCOO make_random_coo(
    int rows, int cols, double density, uint64_t seed) {
    using bsmp::io::parser_utilities::parse_coo;
    Lcg rng(seed);
    std::ostringstream oss;
    const int num_triplets = static_cast<int>(density * rows * cols);
    for (int k = 0; k < num_triplets; ++k) {
        const int i = rng.next_int(0, rows);
        const int j = rng.next_int(0, cols);
        const float v = rng.next_float01() * 2.0f - 1.0f;
        oss << i << ' ' << j << ' ' << v << '\n';
    }
    std::istringstream iss(oss.str());
    return parse_coo(iss);
}

inline bsmp::io::parser_utilities::HostCOO make_random_coo(
    int n, double density, uint64_t seed) {
    return make_random_coo(n, n, density, seed);
}

// ---------------------------------------------------------------------------
// Device <-> host round trips through the public DenseVector API.
// ---------------------------------------------------------------------------
inline bsmp::core::DenseVector to_device(const Vec& v) {
    return bsmp::core::DenseVector(v.begin(), v.end());
}

inline Vec from_device(const bsmp::core::DenseVector& v) {
    Vec out(v.size());
    v.copy_to(out.data());
    return out;
}

// ---------------------------------------------------------------------------
// Assertions. relative-error style with a 1 + |ref| scale factor so near-zero
// rows do not make the test flaky.
// ---------------------------------------------------------------------------
inline void expect_vec_close(const Vec& ref, const Vec& got,
                             double rel_tol = 1e-4) {
    ASSERT_EQ(ref.size(), got.size()) << "vector sizes differ";
    for (size_t i = 0; i < ref.size(); ++i) {
        const double tol = rel_tol * (1.0 + std::fabs(static_cast<double>(ref[i])));
        EXPECT_NEAR(static_cast<double>(got[i]),
                    static_cast<double>(ref[i]), tol)
            << "element " << i;
    }
}

}  // namespace spmv

#endif  // BSMP_TESTS_SPMV_TEST_UTILS_HPP