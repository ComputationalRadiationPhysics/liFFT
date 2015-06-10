#pragma once

#include "foobar/types/Real.hpp"

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

}  // namespace foobar
