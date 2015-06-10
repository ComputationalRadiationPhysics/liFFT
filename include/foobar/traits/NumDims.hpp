#pragma once

#include "foobar/traits/void_t.hpp"

namespace foobar {
namespace traits {

    /**
     * Returns the number of dimensions for the given array-like type
     */
    template< typename T, typename T_SFINAE = void >
    struct NumDims;

    template< typename T >
    struct NumDims< T, void_t< decltype(T::numDims) > >: std::integral_constant< unsigned, T::numDims >{};

}  // namespace traits
}  // namespace foobar
