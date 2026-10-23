#ifndef BSMP_PROF_HOLDER_H
#define BSMP_PROF_HOLDER_H
#include <type_traits>
#include <utility>

namespace bsmp {

namespace prof {

template <class P, bool = std::is_empty_v<P> && !std::is_final_v<P>>
struct InstrHolder;

template <class P>
struct InstrHolder<P, true> : private P {
    P& policy() noexcept { return *this; }
    const P& policy() const noexcept { return *this; }
};

template <class P>
struct InstrHolder<P, false> {
    P& policy() noexcept { return policy_; }
    const P& policy() const noexcept { return policy_; }

   private:
    P policy_{};
};

}  // namespace prof
}  // namespace bsmp

#endif  // BSMP_PROF_HOLDER_H
