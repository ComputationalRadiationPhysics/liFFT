#pragma once

#include "foobar/types/Real.hpp"
#include "foobar/policies/GetRawPtr.hpp"
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

    namespace policies {

        template< typename T >
        struct GetRawPtr< types::RealValues<T> >
        {
            using Data = types::RealValues<T>;
            using Ptr = T*;
            using ConstPtr = const T*;

            Ptr
            operator()(Data& data){
                return &data[0].value;
            }

            ConstPtr
            operator()(const Data& data){
                return &data[0].value;
            }
        };

    }  // namespace policies
}  // namespace foobar
