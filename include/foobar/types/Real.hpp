#pragma once

#include "foobar/traits/IsComplex.hpp"

namespace foobar {
    namespace types {

        template< typename T >
        struct Real
        {
            using type = T;
            static constexpr bool isComplex = false;
            T value;

            Real(){}
            Real(T val):value(val){}
            operator T(){return value;}
        };

    }  // namespace types

    namespace traits {

        template< typename T >
        struct IsComplex< types::Real<T> >: std::false_type{};

    }  // namespace traits

}  // namespace foobar
