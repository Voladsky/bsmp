// Framework quick-start demo: block-COO SpMV on a random matrix.
//
// Pipeline: random triplets -> parse_coo (HostCOO) -> to_coo (COOFormat::Buffers)
//           -> SparseMatrix<COOFormat>::operator* -> host reference comparison.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "core/constants.hpp"
#include "core/dense_vector.hpp"
#include "core/matrix_formats.hpp"
#include "core/sparse_matrix.hpp"
#include "io/parsers_utilities.hpp"
#include "prof/cpu_time_scope.hpp"

int run(int argc, char** argv) {
    using bsmp::core::DenseVector;
    using bsmp::core::SparseMatrix;
    using bsmp::core::formats::COOFormat;
    using bsmp::prof::CPUProfilingContract;
    using namespace bsmp::io::parser_utilities;

    int mults = 10;
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--mults" || arg == "--iterations") && i + 1 < argc) {
            mults = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: framework_spmv_demo [--mults N] [--verbose]\n"
                      << "  --mults N   sequential multiplies (default 10)\n"
                      << "  --verbose   per-iteration timing\n";
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << '\n';
            return 1;
        }
    }

    const int n = 1024;  // element-space matrix size
    const int block = bsmp::core::constants::BLOCK_SIZE;
    const double density = 0.02;
    const unsigned int seed = 12345u;

    // 1. Random element-level triplets ("i j v" lines for parse_coo).
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> row_dist(0, n - 1);
    std::uniform_int_distribution<int> col_dist(0, n - 1);
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);

    std::ostringstream oss;
    const int num_triplets = static_cast<int>(density * n * n);
    for (int k = 0; k < num_triplets; ++k) {
        oss << row_dist(rng) << ' ' << col_dist(rng) << ' ' << val_dist(rng)
            << '\n';
    }

    // 2. Parse into a HostCOO (element triplets -> dense blocks).
    std::istringstream iss(oss.str());
    const HostCOO coo = parse_coo(iss);
    std::cout << "matrix " << n << 'x' << n << " block=" << block
              << " triplets=" << num_triplets
              << " blocks=" << coo.block_rows.size() << '\n';

    // 3. Host reference y_ref = A * x_host.
    std::vector<float> x_host(n);
    std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);
    for (float& v : x_host) {
        v = x_dist(rng);
    }

    std::vector<float> y_ref(n, 0.0f);
    for (size_t b = 0; b < coo.block_rows.size(); ++b) {
        const int brow = coo.block_rows[b];
        const int bcol = coo.block_cols[b];
        const float* data = coo.block_data.data() + b * block * block;
        for (int r = 0; r < block; ++r) {
            const int row = brow * block + r;
            if (row >= n) continue;
            float acc = 0.0f;
            for (int c = 0; c < block; ++c) {
                const int col = bcol * block + c;
                if (col >= n) continue;
                acc += data[r * block + c] * x_host[col];
            }
            y_ref[row] += acc;
        }
    }

    // 4. Build the device matrix (mem::default_allocator()).
    COOFormat::Buffers buffers = to_coo(coo);
    SparseMatrix<COOFormat, CPUProfilingContract> A(std::move(buffers));

    // x_host is dead after the reference computation: move it onto the device.
    DenseVector x(std::make_move_iterator(x_host.begin()),
                  std::make_move_iterator(x_host.end()));

    // 5. Benchmark: average per-multiply wall time. y_bench is reused on
    //    purpose (values accumulate) — only timing matters here.
    DenseVector y_bench(n);
    double total_ms = 0.0;
    for (int iter = 0; iter < mults; ++iter) {
        A.multiply(x, y_bench);
        //const double iter_ms =
        //    std::chrono::duration<double, std::milli>(t1 - t0).count();
        //total_ms += iter_ms;
        //if (verbose) {
        //    std::cout << "Iteration " << (iter + 1) << "/" << mults
        //              << ": " << iter_ms << " ms\n";
        //}
    }
    //std::cout << "Total time: " << total_ms << " ms for " << mults
    //          << " multiplies\n";
    //std::cout << "Average per multiply: " << (total_ms / mults) << " ms\n";

    // 6. Correctness: fresh output (operator* zeroes it), download, compare.
    DenseVector y = A * x;
    std::vector<float> y_gpu(n, 0.0f);
    y.copy_to(y_gpu.data());

    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) {
        num += double(y_gpu[i] - y_ref[i]) * (y_gpu[i] - y_ref[i]);
        den += double(y_ref[i]) * y_ref[i];
    }
    const double rel = std::sqrt(num / den);

    std::cout << "relative residual: " << rel << '\n';
    if (rel < 1e-4) {
        std::cout << "[OK]\n";
        return 0;
    }
    std::cout << "[FAIL]\n";
    return 1;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
