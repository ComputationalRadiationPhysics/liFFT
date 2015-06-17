#pragma once

#include "foobar/traits/IntegralType.hpp"
#include "foobar/traits/IsComplex.hpp"
#include "foobar/traits/IsAoS.hpp"
#include "foobar/policies/GetRawPtr.hpp"
#include "foobar/types/Vec.hpp"

namespace foobar {

    namespace types {

        /**
         * Container used to store data with its meta-data
         */
        template< unsigned T_numDims, class T_Memory, class T_Accessor = typename T_Memory::Accessor, bool T_isStrided=false >
        struct DataContainer
        {
            static constexpr unsigned numDims = T_numDims;
            using Memory = T_Memory;
            using Accessor = T_Accessor;
            static constexpr bool isStrided = T_isStrided;

            Vec< numDims > extents;

            Memory data;
        };

        template< unsigned T_numDims, class T_Memory, class T_Accessor >
        struct DataContainer< T_numDims, T_Memory, T_Accessor, true >: DataContainer< T_numDims, T_Memory, T_Accessor, false >
        {
            static constexpr unsigned numDims = T_numDims;
            using Memory = T_Memory;
            using Accessor = T_Accessor;
            static constexpr bool isStrided = true;

            Vec< numDims > strides;
        };

    }  // namespace types

    namespace traits {

        template< class T >
        struct IntegralTypeImpl< T, void_t< typename T::Memory > >: IntegralType< typename T::Memory >{};

        template< class T >
        struct IsComplex<T, void_t< typename T::Memory > >: IsComplex< typename T::Memory >{};

        template< class T >
        struct IsAoS<T, void_t< typename T::Memory > >: IsAoS< typename T::Memory >{};

    }  // namespace traits

    namespace policies {

        template< unsigned U, class T_Memory, class T_Accessor >
        struct GetRawPtr< types::DataContainer< U, T_Memory, T_Accessor, false > >: GetRawPtr<T_Memory>{
            using GetRawPtrInt = GetRawPtr<T_Memory>;
            using Data = types::DataContainer< U, T_Memory, T_Accessor, false >;
            using type = typename GetRawPtrInt::type;

            type
            operator()(Data& data)
            {
                return GetRawPtrInt::operator()(data.data);
            }

            const type
            operator()(const Data& data)
            {
                return GetRawPtrInt::operator()(data.data);
            }
        };

    }  // namespace policies
}  // namespace foobar
