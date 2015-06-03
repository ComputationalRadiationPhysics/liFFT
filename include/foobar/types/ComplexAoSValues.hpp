#pragma once

#include <boost/utility.hpp>
#include "foobar/types/Complex.hpp"
#include "foobar/policies/all.hpp"

namespace foobar {
    namespace types {

        template< typename T >
        using ComplexAoSValues = Complex<T>*;

    }  // namespace types

    namespace traits {

        template< typename T >
        struct IsAoS< types::ComplexAoSValues<T> >: std::true_type{};

    }  // namespace traits

    namespace policies {

        template< typename T >
        struct GetIntensity< types::ComplexAoSValues<T> >
        {
            T operator()(const types::ComplexAoSValues<T>& values, unsigned idx){
                T r = values[idx].real;
                T i = values[idx].imag;
                return r*r + i*i;
            }
        };

        template< typename T >
        struct GetValue< types::ComplexAoSValues<T> >
        {
            using type = types::ComplexAoSValues<T>;

            T getReal(const type& values, unsigned idx){
                return values[idx].real;
            }

            T getImag(const type& values, unsigned idx){
                return values[idx].imag;
            }
        };

        template< typename T >
        struct GetRawPtr< types::ComplexAoSValues<T> >: private boost::noncopyable
        {
            using type = types::ComplexAoSValues<T>;

            GetRawPtr(const type& data): data_(&const_cast<type&>(data)[0].real.value){}

            T*
            operator()(){
                return data_;
            }

        private:
            T* data_;
        };

    }  // namespace policies
}  // namespace foobar
