#pragma once

#include "foobar/traits/IsComplex.hpp"
#include "foobar/traits/IntegralType.hpp"

namespace foobar {
namespace policies {

    namespace detail
    {
        template< class T_GetValue, bool T_isComplex = traits::IsComplex< typename T_GetValue::type >::value >
        struct CalcIntensityImpl
        {
            using RetType = typename traits::IntegralType<T_GetValue>::type;

            template< typename T_Data >
            RetType operator()(const T_Data& data, unsigned idx)
            {
                T_GetValue accessor;
                RetType real = accessor.getReal(data, idx);
                RetType imag = accessor.getImag(data, idx);
                return real*real + imag*imag;
            }
        };

        template< class T_GetValue>
        struct CalcIntensityImpl< T_GetValue, false >
        {
            using RetType = typename traits::IntegralType<T_GetValue>::type;

            template< typename T_Data >
            RetType operator()(const T_Data& data, unsigned idx)
            {
                T_GetValue accessor;
                RetType real = accessor.getReal(data, idx);
                return real*real;
            }
        };

    }//namespace detail

    template< class T_GetValue >
    struct CalcIntensityImpl : public detail::CalcIntensityImpl<T_GetValue>{};

}  // namespace policies
}  // namespace foobar
