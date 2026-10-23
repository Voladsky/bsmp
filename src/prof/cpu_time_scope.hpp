#include <chrono>
#include <iostream>

#include "prof_contract.hpp"

namespace bsmp {

namespace prof {

struct CPUProfilingContract {
    constexpr static bool enabled = true;
    struct Scope {
        std::chrono::steady_clock::time_point start;
        void stop() noexcept {
            auto end = std::chrono::steady_clock::now();
            double iter_ms =
                std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                      start)
                    .count() /
                1000.0;
            std::cout << iter_ms << " ms" << std::endl;
        }
    };
    Scope makeScope() const {
        auto scope = Scope();
        scope.start = std::chrono::steady_clock::now();
        return scope;
    }
};

static_assert(is_profiling_contract<CPUProfilingContract>(),
              "CPUProfilingContract does not satisfy the profiling contract");

}  // namespace prof
}  // namespace bsmp
