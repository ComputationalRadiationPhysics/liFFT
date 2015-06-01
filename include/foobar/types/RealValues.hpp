#pragma once

#include <boost/utility.hpp>
#include "foobar/types/Real.hpp"
#include "foobar/policies/all.hpp"

namespace foobar {
    namespace types {

        template< typename T >
        using RealValues = Real<T>*;

    }  // namespace types

    namespace policies {

        template< typename T >
        struct GetIntensity< types::RealValues<T> >
        {
            T operator()(const types::RealValues<T>& values, unsigned idx){
                T r = values[idx];
                return r*r;
            }
        };

        template< typename T >
        struct GetValue< types::RealValues<T> >
        {
            using type = types::RealValues<T>;

            T getReal(const type& values, unsigned idx){
                return values[idx];
            }
        };

        template< typename T >
        struct GetRawPtr< types::RealValues<T> >: private boost::noncopyable
        {
            using type = types::RealValues<T>;

            GetRawPtr(const type& data): data_(&const_cast<type&>(data)[0].value){}

            T* getData(){
                return data_;
            }

        private:
            T* data_;
        };

    }  // namespace policies
}  // namespace foobar
