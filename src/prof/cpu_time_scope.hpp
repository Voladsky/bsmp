#ifndef BSMP_CPU_TIME_SCOPE_H
#define BSMP_CPU_TIME_SCOPE_H

#include <chrono>
#include <iostream>

#include "prof_contract.hpp"
#include "spmv_metrics.hpp"

namespace bsmp {

namespace prof {

struct CPUProfilingContract {
    static constexpr bool enabled = true;
    struct Scope {
        Scope(const CPUProfilingContract* owner) : owner_(owner) {}
        std::chrono::steady_clock::time_point start;
        void stop() noexcept {
            auto end = std::chrono::steady_clock::now();
            double iter_ms =
                std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                      start)
                    .count() /
                1000.0;
            owner_->set_last(make_metrics(iter_ms));
            return;
        }

       private:
        const CPUProfilingContract* owner_;
    };
    Scope makeScope() const {
        auto scope = Scope(this);
        scope.start = std::chrono::steady_clock::now();
        return scope;
    }

    const SpmvMetrics& last() const noexcept { return last_; }
    void set_last(const SpmvMetrics& m) const noexcept { last_ = m; }

   private:
    mutable SpmvMetrics last_{};
};

static_assert(is_profiling_contract<CPUProfilingContract>(),
              "CPUProfilingContract does not satisfy the profiling contract");

}  // namespace prof
}  // namespace bsmp

#endif  // BSMP_CPU_TIME_SCOPE_H
