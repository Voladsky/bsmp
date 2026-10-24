#ifndef BSMP_PROF_CONTRACT
#define BSMP_PROF_CONTRACT

#include <type_traits>
#include <utility>

namespace bsmp {

namespace prof {

template <class, class = void>
struct is_profiling_contract : std::false_type {
};  // if void_t couldn't instantiate something, default to this

template <class T>
struct is_profiling_contract<
    T, std::void_t<typename T::Scope,
                   decltype(std::declval<const T&>().makeScope()),
                   decltype(std::declval<typename T::Scope&>().stop()),
                   decltype(T::enabled)>>
    : std::integral_constant<
          bool, std::is_same_v<decltype(std::declval<const T&>().makeScope()),
                               typename T::Scope> &&
                    noexcept(std::declval<typename T::Scope&>().stop())> {};
}  // namespace prof
}  // namespace bsmp

#endif BSMP_PROF_CONTRACT
