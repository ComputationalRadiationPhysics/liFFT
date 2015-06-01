#pragma once

#include "foobar/traits/IsComplex.hpp"

namespace foobar {
namespace traits {

    /**
     * Evaluates to a true type if the given complex type is an Array of Structs, false otherwise
     * Defaults to false for Real values
     */
    template< typename T, typename T_IsComplex = IsComplex<T> >
    struct IsAoS: std::false_type{};

    /**
     * Undefined for (unknown) complex types
     */
    template< typename T >
    struct IsAoS< T, std::true_type >;

}  // namespace traits
}  // namespace foobar
