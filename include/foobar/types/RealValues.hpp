#pragma once

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
#include <boost/utility.hpp>
        template< typename T >
        struct GetValue< types::RealValues<T> >
        {
            using type = types::RealValues<T>;

            T getReal(const type& values, unsigned idx){
                return values[idx];
            }
        };

        template< typename T >
        struct GetRawPtr< types::RealValues<T> >
        {
            using Data = types::RealValues<T>;
            using type = T*;

            type
            operator()(Data& data){
                return &data[0].value;
            }

            const type
            operator()(const Data& data){
                return &data[0].value;
            }
        };

    }  // namespace policies
}  // namespace foobar
