#pragma once

#include <type_traits>

namespace std {

    template< bool B, class T = void >
    using enable_if_t = typename enable_if<B,T>::type;

}  // namespace std
