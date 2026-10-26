#ifndef BSMP_PROF_HOLDER_H
#define BSMP_PROF_HOLDER_H
#include <type_traits>
#include <utility>

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

}  // namespace prof
}  // namespace bsmp

#endif  // BSMP_PROF_HOLDER_H
