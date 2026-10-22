#include "prof_contract.hpp"

namespace bsmp {

namespace prof {

struct NoProfilingContract {
    constexpr static bool enabled = false;
    struct Scope {
        void stop() noexcept {}
    };
    Scope makeScope() const {}
};

static_assert(is_profiling_contract<NoProfilingContract>(), "NoProfilingContract does not satisfy the profiling contract");

}  // namespace prof
}  // namespace bsmp
