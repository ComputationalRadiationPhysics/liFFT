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

    template<>
    struct IsComplex< double >: std::false_type{};

    template<>
    struct IsComplex< float >: std::false_type{};

}  // namespace traits
}  // namespace foobar
