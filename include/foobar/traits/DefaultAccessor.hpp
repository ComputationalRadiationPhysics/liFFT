#pragma once

#include "foobar/policies/ArrayAccessor.hpp"

namespace foobar {
namespace traits {

    /**
     * Returns the default accessor for the given type
     */
    template< typename T, typename T_SFINAE = void >
    struct DefaultAccessor{
        using type = typename T::Accessor;
    };

    /**
     * Specialization for pointer types
     */
    template< typename T >
    struct DefaultAccessor<T*>{
        using type = policies::ArrayAccessor<false>;
    };

    template< typename T >
    using DefaultAccessor_t = typename DefaultAccessor<T>::type;

}  // namespace traits
}  // namespace foobar
