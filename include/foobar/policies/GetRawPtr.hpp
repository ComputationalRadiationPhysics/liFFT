#pragma once

namespace foobar {
namespace policies {

    /**
     * Returns the raw ptr to the underlying (internal) data
     * This will be a pointer for a single pointer or a pair for ComplexSoA
     * Takes the object in the ()-operator
     * It should also declare the type member ::type for the return type
     */
    template< typename T_Memory >
    struct GetRawPtr;

    template< typename T >
    struct GetRawPtr< T* >
    {
        static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value, "You must provide an own GetRawPt for types other than pointers to integral/floating point types");
        using type = T*;

        type
        operator()(type data){
            return data;
        }
    };

    template< typename T >
    struct GetRawPtr< T[] >: GetRawPtr< T* >
    {
        static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value, "You must provide an own GetRawPt for types other than pointers to integral/floating point types");
    };

}  // namespace policies
}  // namespace foobar
