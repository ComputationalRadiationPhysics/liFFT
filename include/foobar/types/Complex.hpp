#pragma once

#include "foobar/types/Real.hpp"
#include "foobar/traits/IsComplex.hpp"

namespace foobar {
    namespace types {

        template< typename T=double >
        struct Complex
        {
            using type = T;
            static constexpr bool isComplex = true;
            Real<T> real, imag;
        };

    }  // namespace types

    namespace traits {

        template< typename T >
        struct IsComplex< types::Complex<T> >: std::true_type{};

    }  // namespace traits

}  // namespace foobar
