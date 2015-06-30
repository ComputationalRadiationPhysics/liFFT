#pragma once

#include "foobar/traits/IsComplex.hpp"

namespace foobar {
namespace policies {

    struct CalcIntensityFunc
    {
        template< typename T, typename = std::enable_if_t< foobar::traits::IsComplex<T>::value > >
        auto
        operator()(const T& val) const
        -> decltype(val.real*val.real + val.imag*val.imag)
        {
            return val.real*val.real + val.imag*val.imag;
        }

        template< typename T, typename = std::enable_if_t< foobar::traits::IsComplex<T>::value > >
        auto
        operator()(const T& val) const
        -> decltype(val[0]*val[0] + val[1]*val[1])
        {
            return val[0]*val[0] + val[1]*val[1];
        }

        template< typename T, typename = std::enable_if_t< !foobar::traits::IsComplex<T>::value > >
        auto
        operator()(const T& val) const
        -> decltype(val*val)
        {
            return val*val;
        }
    };

}  // namespace policies
}  // namespace foobar
