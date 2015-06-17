#pragma once

#include <memory>
#include "foobar/types/Complex.hpp"
#include "foobar/policies/GetRawPtr.hpp"
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

    namespace policies {

        template< typename T >
        struct GetRawPtr< types::ComplexAoSValues<T> >
        {
            using Data = types::ComplexAoSValues<T>;
            using type = T*;

            type
            operator()(Data& data){
                return &data[0].real.value;
            };

            const type
            operator()(const Data& data){
                return &data[0].real.value;
            };
        };

    }  // namespace policies
}  // namespace foobar
