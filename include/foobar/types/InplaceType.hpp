#pragma once

#include "foobar/traits/IsInplace.hpp"

namespace foobar {
    namespace types {

        template< bool T_isComplex = true >
        struct InplaceType{
            static constexpr bool isComplex = T_isComplex;
        };

    }  // namespace types

    namespace traits {

        template< bool T_isComplex >
        struct IsInplace< types::InplaceType<T_isComplex> >: std::true_type{};

    }  // namespace traits

}  // namespace foobar
