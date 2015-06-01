#pragma once

#include "foobar/traits/IntegralType.hpp"
#include "foobar/policies/GetValue.hpp"
#include "foobar/policies/GetRawPtr.hpp"
#include "foobar/types/Vec.hpp"

namespace foobar {

    namespace types {

        template< unsigned T_numDims, class T_Memory, bool T_isStrided=false >
        struct InputData
        {
            static constexpr unsigned numDims = T_numDims;
            using Memory = T_Memory;
            static constexpr bool isStrided = T_isStrided;

            Vec< unsigned, numDims > extents;
            typename std::conditional< T_isStrided, Vec< unsigned, numDims >, ConstVec< unsigned, 1, numDims > >::type strides;

            Memory data;
        };

    }  // namespace types

    namespace traits {

        template< unsigned U, class T_Memory, bool V >
        struct IntegralType< types::InputData< U, T_Memory, V > >: IntegralType<T_Memory>{};

    }  // namespace traits

    namespace policies {

        template< unsigned U, class T_Memory, bool V >
        struct GetValue< types::InputData< U, T_Memory, V > >{
            using Data = types::InputData< U, T_Memory, V >;
            using GetValueInt = GetValue<T_Memory>;
            using type = typename GetValueInt::type;

            GetValueInt getValue;

            typename traits::IntegralType<T_Memory>::type
            getReal(const Data& values, unsigned idx){
                return getValue.getReal(values.data, idx);
            }

            template< typename = std::enable_if< traits::IsComplex<type>::value > >
            typename traits::IntegralType<T_Memory>::type
            getImag(const Data& values, unsigned idx){
                return getValue.getImag(values.data, idx);
            }
        };

        template< unsigned U, class T_Memory >
        struct GetRawPtr< types::InputData< U, T_Memory, false > >: GetRawPtr<T_Memory>{
            using Data = types::InputData< U, T_Memory, false >;
            using GetRawPtrInt = GetRawPtr<T_Memory>;

            GetRawPtr(const Data& data): GetRawPtrInt(data.data){}
        };

    }  // namespace policies
}  // namespace foobar
