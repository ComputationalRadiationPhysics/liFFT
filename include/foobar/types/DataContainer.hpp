#pragma once

#include "foobar/traits/IntegralType.hpp"
#include "foobar/traits/IsComplex.hpp"
#include "foobar/traits/IsAoS.hpp"
#include "foobar/policies/GetValue.hpp"
#include "foobar/policies/GetRawPtr.hpp"
#include "foobar/policies/GetIntensity.hpp"
#include "foobar/types/Vec.hpp"

namespace foobar {

    namespace types {

        /**
         * Container used to store data with its meta-data
         */
        template< unsigned T_numDims, class T_Memory, bool T_isStrided=false >
        struct DataContainer
        {
            static constexpr unsigned numDims = T_numDims;
            using Memory = T_Memory;
            static constexpr bool isStrided = T_isStrided;

            Vec< numDims > extents;

            Memory data;
        };

        template< unsigned T_numDims, class T_Memory >
        struct DataContainer< T_numDims, T_Memory, true >: DataContainer< T_numDims, T_Memory, false >
        {
            static constexpr unsigned numDims = T_numDims;
            using Memory = T_Memory;
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

        template< unsigned U, class T_Memory, bool V >
        struct GetValue< types::DataContainer< U, T_Memory, V > >{
            using Data = types::DataContainer< U, T_Memory, V >;
            using GetValueInt = GetValue<T_Memory>;
            using type = typename GetValueInt::type;

            GetValueInt getValue;

            typename traits::IntegralType<T_Memory>::type
            getReal(const Data& values, unsigned idx){
                return getValue.getReal(values.data, idx);
            }

            template< class T_Type = type, typename = std::enable_if_t< traits::IsComplex<T_Type>::value > >
            typename traits::IntegralType<T_Memory>::type
            getImag(const Data& values, unsigned idx){
                return getValue.getImag(values.data, idx);
            }
        };

        template< unsigned U, class T_Memory >
        struct GetRawPtr< types::DataContainer< U, T_Memory, false > >: GetRawPtr<T_Memory>{
            using GetRawPtrInt = GetRawPtr<T_Memory>;
            using Data = types::DataContainer< U, T_Memory, false >;
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

        template< unsigned u, class T_Memory, bool v >
        struct GetIntensity< types::DataContainer< u, T_Memory, v > >{
            GetIntensity<T_Memory> getIntensity;

            template< typename T_Arg >
            auto
            operator()(T_Arg&& values, unsigned idx)
            -> decltype( getIntensity(std::forward<T_Arg>(values), idx) )
            {
                return getIntensity(std::forward<T_Arg>(values), idx);
            }
        };

    }  // namespace policies
}  // namespace foobar
