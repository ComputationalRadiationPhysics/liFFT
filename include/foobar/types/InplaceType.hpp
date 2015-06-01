#pragma once

#include "foobar/traits/IsInplace.hpp"
#include "foobar/traits/IsComplex.hpp"

namespace foobar {
    namespace types {

        template< bool T_isComplex = true >
        struct InplaceType{};

    }  // namespace types

    namespace traits {

        template< bool T_isComplex >
        template
        struct IsInplace< types::InplaceType<T_isComplex> >: std::true_type{};

        template< bool T_isComplex >
        struct IsComplex< types::InplaceType<T_isComplex> >: std::integral_constant< bool, T_isComplex >{};

    }  // namespace traits

}  // namespace foobar
