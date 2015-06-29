#pragma once

#include "foobar/traits/IntegralType.hpp"
#include "foobar/traits/NumDims.hpp"
#include "foobar/traits/IsComplex.hpp"
#include "foobar/traits/IsStrided.hpp"
#include "foobar/traits/IsAoS.hpp"
#include "foobar/policies/GetExtents.hpp"
#include "foobar/policies/GetExtentsRawPtr.hpp"

namespace foobar {
    namespace types {

        /**
         * Wraps a type such that all dimensions are shifted by T_dimOffset effectively reducing the number of dimensions
         * That is for a 3D type with a dimOffset of 1 only the last 2 dimensions will be available
         * The first (slowest varying) dimension is stripped
         */
        template< class T_Base, unsigned T_dimOffset >
        class DimOffsetWrapper: public T_Base
        {
        public:
            using Base = T_Base;
            static constexpr unsigned dimOffset = T_dimOffset;

            static constexpr unsigned origNumDims = traits::NumDims<Base>::value;
            static_assert(dimOffset < origNumDims, "Cannot remove all dimensions");
            static constexpr unsigned numDims = origNumDims - dimOffset;

            using Base::Base;
        };

    }  // namespace types

    namespace traits {

        template< class T_Base, unsigned T_dimOffset >
        struct IntegralType< types::DimOffsetWrapper< T_Base, T_dimOffset > >: IntegralType<T_Base>{};

        template< class T_Base, unsigned T_dimOffset >
        struct IsComplex< types::DimOffsetWrapper< T_Base, T_dimOffset > >: IsComplex<T_Base>{};

        template< class T_Base, unsigned T_dimOffset >
        struct IsStrided< types::DimOffsetWrapper< T_Base, T_dimOffset > >: IsStrided<T_Base>{};

        template< class T_Base, unsigned T_dimOffset >
        struct IsAoS< types::DimOffsetWrapper< T_Base, T_dimOffset > >: IsAoS<T_Base>{};

    }  // namespace traits

    namespace policies {

        template< class T_Base, unsigned T_dimOffset >
        struct GetExtents< types::DimOffsetWrapper< T_Base, T_dimOffset > >: GetExtents< T_Base >
        {
            using Parent = GetExtents< T_Base >;
            using Parent::Parent;

            unsigned operator[](unsigned dimIdx) const
            {
                return Parent::operator[](dimIdx + T_dimOffset);
            }
        };

        template< class T_Base, unsigned T_dimOffset >
        struct GetExtentsRawPtr< types::DimOffsetWrapper< T_Base, T_dimOffset > >: GetExtentsRawPtr< T_Base >
        {
            using Parent = GetExtentsRawPtr< T_Base >;
            using Parent::Parent;

            const unsigned* operator()() const
            {
                return Parent::operator()() + T_dimOffset;
            }
        };

    }  // namespace policies
}  // namespace foobar
