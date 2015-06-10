#pragma once

#include "foobar/types/RealValues.hpp"
#include "foobar/traits/IsAoS.hpp"
#include "foobar/policies/all.hpp"

namespace foobar {
    namespace types {

        template< typename T >
        struct ComplexSoAValues
        {
            using type = T;
            static constexpr bool isComplex = true;
            RealValues<T> real, imag;
        };

    }  // namespace types

    namespace traits {

        template< typename T >
        struct IsAoS< types::ComplexSoAValues<T> >: std::false_type{};

    }  // namespace traits

    namespace policies {

        template< typename T >
        struct GetIntensity< types::ComplexSoAValues<T> >
        {
            T operator()(const types::ComplexSoAValues<T>& values, unsigned idx){
                T r = values.real[idx];
                T i = values.imag[idx];
                return r*r + i*i;
            }
        };

        template< typename T >
        struct GetValue< types::ComplexSoAValues<T> >
        {
            using type = types::ComplexSoAValues<T>;

            T getReal(const type& values, unsigned idx){
                return values.real[idx];
            }

            T getImag(const type& values, unsigned idx){
                return values.imag[idx];
            }
        };

        template< typename T >
        struct GetRawPtr< types::ComplexSoAValues<T> >
        {
            using Data = types::ComplexSoAValues<T>;
            using type = std::pair< T*, T* >;

            type
            operator()(Data& data){
                return std::make_pair(&data.real[0].value, &data.imag[0].value);
            }

            const type
            operator()(const Data& data){
                return std::make_pair(&data.real[0].value, &data.imag[0].value);
            }
        };


    }  // namespace policies
}  // namespace foobar
