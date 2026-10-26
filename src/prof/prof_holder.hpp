#ifndef BSMP_PROF_HOLDER_H
#define BSMP_PROF_HOLDER_H
#include <type_traits>
#include <utility>

#include "spmv_metrics.hpp"

namespace bsmp {

namespace prof {

template <class P, bool = std::is_empty_v<P> && !std::is_final_v<P>>
struct ProfHolder;

template <class P>
struct ProfHolder<P, true> : private P {
    static constexpr bool enabled = P::enabled;
    P& policy() noexcept { return *this; }
    const P& policy() const noexcept { return *this; }
};

template <class P>
struct ProfHolder<P, false> {
    static constexpr bool enabled = P::enabled;
    P& policy() noexcept { return policy_; }
    const P& policy() const noexcept { return policy_; }

   private:
    P policy_{};
};

// Conditionally exposes lastMetrics() to the owning class.
//
// Presence is decided by partial specialization of the base class rather than
// by SFINAE on a member of the owning template: MSVC eagerly instantiates
// member-template declarations (default template arguments and trailing
// return types) when a class template is instantiated, turning
// enable_if_t<false, ...> into a hard C2938 error. With base-class selection
// the disabled case never even declares the member.
//
// The default argument is taken from ProfHolder<P>::enabled (a direct public
// member of the class) rather than P::enabled: inside SparseMatrix — which
// privately inherits ProfHolder<P>, which privately inherits P — MSVC
// resolves P::enabled through the inheritance path and trips C2247.
// Qualified lookup into ProfHolder's own scope does not.
template <class P, bool = ProfHolder<P>::enabled>
struct MetricsAccessor {};

template <class P>
struct MetricsAccessor<P, false> {
    explicit MetricsAccessor(const P*) noexcept {}
    void rewire(const P*) noexcept {}
};

template <class P>
struct MetricsAccessor<P, true> {
    explicit MetricsAccessor(const P* policy) noexcept : policy_(policy) {}
    // Re-point after the owner has been moved; the policy subobject lives
    // inside the owner, so its address changes.
    void rewire(const P* policy) noexcept { policy_ = policy; }
    const SpmvMetrics& lastMetrics() const noexcept { return policy_->last(); }

   private:
    const P* policy_;
};

}  // namespace prof
}  // namespace bsmp

#endif  // BSMP_PROF_HOLDER_H
