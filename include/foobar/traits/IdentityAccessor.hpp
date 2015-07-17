#pragma once

#include "foobar/accessors/ArrayAccessor.hpp"

namespace foobar {
namespace traits {

    /**
     * Returns the default accessor for the given type
     */
    template< typename T, typename T_SFINAE = void >
    struct IdentityAccessor{
        using type = typename T::IdentityAccessor;
    };

    /**
     * Specialization for pointer types
     */
    template< typename T >
    struct IdentityAccessor<T*>{
        using type = accessors::ArrayAccessor<false>;
    };

    /**
     * Specialization for reference types
     */
    template< typename T >
    struct IdentityAccessor<T&>: IdentityAccessor<T>{};

    /**
     * Specialization for const types
     */
    template< typename T >
    struct IdentityAccessor<const T>: IdentityAccessor<T>{};

    template< typename T >
    using IdentityAccessor_t = typename IdentityAccessor<T>::type;

    /**
     * Returns an instance of the default accessor for the given container
     *
     * @param The container to get the default accessor for
     * @return Instance of an accessor
     */
    template< typename T >
    IdentityAccessor_t<T>
    getIdentityAccessor(const T&){
        return IdentityAccessor_t<T>();
    }

}  // namespace traits
}  // namespace foobar
