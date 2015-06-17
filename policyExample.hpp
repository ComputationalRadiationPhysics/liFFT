#pragma once

#include "foobar/traits/IntegralType.hpp"
#include "foobar/traits/IsComplex.hpp"
#include "foobar/traits/IsAoS.hpp"
#include "foobar/policies/GetExtents.hpp"
#include "foobar/policies/GetStrides.hpp"
#include "foobar/policies/DataContainerAccessor.hpp"
#include "foobar/policies/CalcIntensity.hpp"

namespace foobar {

    template< class T_Input >
    std::enable_if_t< (traits::IsComplex<T_Input>::value && traits::IsAoS<T_Input>::value) >
    calcIntensity(T_Input& input, typename traits::IntegralType<T_Input>::type* output)
    {
        policies::GetExtents<T_Input> extents(input);
        policies::GetStrides<T_Input> strides(input);
        for(unsigned i=0; i<extents[0]; ++i){
            unsigned idx = i*strides[0];
            auto real = input.data[idx].real;
            auto imag = input.data[idx].imag;
            output[i] = real*real+imag*imag;
        }
    }

    template< class T_Input >
    std::enable_if_t< (traits::IsComplex<T_Input>::value && !traits::IsAoS<T_Input>::value) >
    calcIntensity(T_Input& input, typename traits::IntegralType<T_Input>::type* output)
    {
        policies::GetExtents<T_Input> extents(input);
        policies::GetStrides<T_Input> strides(input);
        for(unsigned i=0; i<extents[0]; ++i){
            unsigned idx = i*strides[0];
            auto real = input.data.getRealData()[idx];
            auto imag = input.data.getImagData()[idx];
            output[i] = real*real+imag*imag;
        }
    }

    template< class T_Input >
    std::enable_if_t< !traits::IsComplex<T_Input>::value >
    calcIntensity(T_Input& input, typename traits::IntegralType<T_Input>::type* output)
    {
        policies::GetExtents<T_Input> extents(input);
        policies::GetStrides<T_Input> strides(input);
        for(unsigned i=0; i<extents[0]; ++i){
            unsigned idx = i*strides[0];
            auto real = input.data[idx];
            output[i] = real*real;
        }
    }

    template< class T_Input >
    void
    calcIntensity2(T_Input& input, typename traits::IntegralType<T_Input>::type* output)
    {
        policies::CalcIntensity< policies::DataContainerAccessor > calcIntensity;
        policies::GetExtents<T_Input> extents(input);
        policies::GetStrides<T_Input> strides(input);
        for(unsigned i=0; i<extents[0]; ++i){
            unsigned idx = i*strides[0];
            output[i] = calcIntensity(input, idx);
        }
    }



}  // namespace foobar
