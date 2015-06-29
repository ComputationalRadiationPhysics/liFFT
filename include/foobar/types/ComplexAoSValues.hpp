#pragma once

#include <memory>
#include "foobar/types/Complex.hpp"
#include "foobar/types/AoSValues.hpp"

namespace foobar {
    namespace types {

        template< typename T, bool T_ownsPointer = true >
        class ComplexAoSValues: public detail::AoSValues< Complex<T>, T_ownsPointer >
        {
        public:
            using Parent = detail::AoSValues< Complex<T>, T_ownsPointer >;

            using Parent::Parent;
        };

    }  // namespace types

}  // namespace foobar
