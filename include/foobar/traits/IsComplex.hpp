#pragma once

#include "foobar/void_t.hpp"

namespace foobar {
namespace traits {

    /**
     * Evaluates to true type if the given type is Complex, false for Real
     */
    template< typename T, typename T_SFINAE = void >
    struct IsComplex;

    /**
     * Specialization for pointers
     */
    template< typename T >
    struct IsComplex< T* >: IsComplex< T >{};

    template< typename T >
    struct IsComplex< T& >: IsComplex< T >{};

    template<>
    struct IsComplex< double >: std::false_type{};

    template<>
    struct IsComplex< float >: std::false_type{};

    template< typename T >
    struct IsComplex< T, void_t< decltype(T::isComplex) > >: std::integral_constant< bool, T::isComplex >{};

}  // namespace traits
}  // namespace foobar
