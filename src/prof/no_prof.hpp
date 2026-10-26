#ifndef BSMP_NO_PROF_HPP
#define BSMP_NO_PROF_HPP

#include "prof_contract.hpp"

namespace bsmp {

namespace prof {

struct NoProfilingContract {
    constexpr static bool enabled = false;
    struct Scope {
        void stop() noexcept {}
    };
    Scope makeScope() const { return {}; }
};

static_assert(is_profiling_contract<NoProfilingContract>(), "NoProfilingContract does not satisfy the profiling contract");

}  // namespace prof
}  // namespace bsmp

#endif // BSMP_NO_PROF_HPP
