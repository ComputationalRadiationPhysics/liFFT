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

    /**
     * Specialization for reference types
     */
    template< typename T >
    struct DefaultAccessor<T&>: DefaultAccessor<T>{};

    /**
     * Specialization for const types
     */
    template< typename T >
    struct DefaultAccessor<const T>: DefaultAccessor<T>{};

    template< typename T >
    using DefaultAccessor_t = typename DefaultAccessor<T>::type;

    /**
     * Returns an instance of the default accessor for the given container
     *
     * @param The container to get the default accessor for
     * @return Instance of an accessor
     */
    template< typename T >
    DefaultAccessor_t<T>
    getDefaultAccessor(const T&){
        return DefaultAccessor_t<T>();
    }

}  // namespace traits
}  // namespace foobar
