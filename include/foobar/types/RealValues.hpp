#pragma once

#include "foobar/types/Real.hpp"
#include "foobar/types/AoSValues.hpp"

namespace foobar {
    namespace types {

        template< typename T, bool T_ownsPointer = true >
        class RealValues: public detail::AoSValues< Real<T>, T_ownsPointer >
        {
        public:
            using Parent = detail::AoSValues< Real<T>, T_ownsPointer >;

            using Parent::Parent;
        };

    }  // namespace types

}  // namespace foobar
