#pragma once

#include "foobar/traits/void_t.hpp"

namespace foobar {
namespace traits {

    /**
     * Evaluates to true type if the given Array-like type is strided
     */
    template< typename T, typename T_SFINAE = void >
    struct IsStrided;

    template< typename T >
    struct IsStrided< T, void_t< decltype(T::isStrided) > >: std::integral_constant< bool, T::isStrided >{};

}  // namespace traits
}  // namespace foobar
