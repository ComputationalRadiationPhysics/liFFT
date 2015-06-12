#pragma once

#include "foobar/traits/IsComplex.hpp"

namespace foobar {
namespace traits {

    /**
     * Evaluates to a true type if the given complex type is an Array of Structs, false otherwise
     * Defaults to false for real values
     */
    template< typename T, typename T_SFINAE = void >
    struct IsAoS: std::false_type
    {
        static_assert(!IsComplex<T>::value, "A user specialization must be provided for Complex types");
    };


    template< typename T >
    struct IsAoS< T, void_t< decltype(T::isAoS) > >: std::integral_constant< bool, T::isAoS >{};


}  // namespace traits
}  // namespace foobar
