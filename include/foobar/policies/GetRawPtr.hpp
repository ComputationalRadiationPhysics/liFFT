#pragma once

namespace foobar {
namespace policies {

    /**
     * Returns the raw ptr to the underlying data
     * This will be a pointer for a single pointer or a pair for ComplexSoA
     * Takes the object in the constructor and defines the ()-operator
     */
    template< typename T_Memory >
    struct GetRawPtr;

    template< typename T >
    struct GetRawPtr< T* >
    {
        static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value, "You must provide an own GetRawPt for types other than pointers to integral/floating point types");
        using type = T*;

        GetRawPtr(const type& data): data_(data){}

        type
        operator()(){
            return data_;
        }

    private:
        type data_;
    };

    template< typename T >
    struct GetRawPtr< T[] >
    {
        static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value, "You must provide an own GetRawPt for types other than pointers to integral/floating point types");
        using type = T*;

        GetRawPtr(const type& data): data_(data){}

        type
        operator()(){
            return data_;
        }

    private:
        type data_;
    };

}  // namespace policies
}  // namespace foobar
