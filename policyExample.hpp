#pragma once

#include <array>
#include <type_traits>
#include "foobar/traits/all.hpp"
#include "foobar/types/all.hpp"
#include "foobar/policies/all.hpp"

namespace foobar {

    template< class T_Input >
    void
    calcIntensity(T_Input& input, typename traits::IntegralType<T_Input>::type* output)
    {
        policies::GetIntensity< T_Input > getIntensity;
        policies::GetExtents<T_Input> extents(input);
        policies::GetStrides<T_Input> strides(input);
        for(unsigned i=0; i<extents[0]; ++i){
            output[i] = getIntensity(input.data, i*strides[0]);
        }
    }

    template< class T_Input >
    void
    calcIntensity2(T_Input& input, typename traits::IntegralType<T_Input>::type* output)
    {
        policies::CalcIntensityImpl< policies::GetValue<T_Input> > calcIntensity;
        policies::GetExtents<T_Input> extents(input);
        policies::GetStrides<T_Input> strides(input);
        for(unsigned i=0; i<extents[0]; ++i){
            unsigned idx = i*strides[0];
            output[i] = calcIntensity(input, idx);
        }
    }



}  // namespace foobar
