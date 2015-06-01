#pragma once

namespace foobar {
namespace traits {

    /**
     * Evaluates to true type if the given type is Complex, false for Real
     */
    template< typename T >
    struct IsComplex;

    /**
     * Specialization for pointers
     */
    template< typename T >
    struct IsComplex< T* >: IsComplex< T >{};

}  // namespace traits
}  // namespace foobar
