#pragma once

#include <type_traits>

namespace std {

    template< bool B, class T = void >
    using enable_if_t = typename enable_if<B,T>::type;

    template< bool B, class T, class U >
    using conditional_t = typename conditional< B, T, U >::type;

    template< class T >
    using result_of_t = typename result_of<T>::type;

}  // namespace std
