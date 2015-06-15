#pragma once

#include "foobar/traits/IsComplex.hpp"

namespace foobar {
namespace traits {

    /**
     * Evaluates to a true type if the given complex type is an Array of Structs
     * (that is the data has only 1 pointer and complex values are interleaved),
     * false otherwise (you have 2 pointers for real and imaginary data)
     * Defaults to true for real values
     */
    template< typename T, typename T_SFINAE = void >
    struct IsAoS: std::true_type
    {
        static_assert(!IsComplex<T>::value, "A user specialization must be provided for Complex types");
    };


    template< typename T >
    struct IsAoS< T, void_t< decltype(T::isAoS) > >: std::integral_constant< bool, T::isAoS >{};


}  // namespace traits
}  // namespace foobar
