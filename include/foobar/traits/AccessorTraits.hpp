#pragma once

#include "foobar/void_t.hpp"
#include "foobar/traits/NumDims.hpp"
#include "foobar/c++14_types.hpp"

namespace foobar {
namespace traits {

    template<
        class T_Accessor,
        typename T_Data,
        typename T_Index = types::Vec< NumDims<T_Data>::value >,
        typename T_SFINAE = void
    >
    struct IsReadAccessor: std::false_type{};

    template< class T_Accessor, typename T_Data, typename T_Index>
    struct IsReadAccessor< T_Accessor, T_Data, T_Index, void_t<
            std::result_of_t<T_Accessor(T_Index, T_Data)>
        >
    >: std::true_type{};

    template<
        class T_Accessor,
        typename T_Data,
        typename T_Value,
        typename T_Index = types::Vec< NumDims<T_Data>::value >,
        typename T_SFINAE = void
    >
    struct IsWriteAccessor: std::false_type{};

    template< class T_Accessor, typename T_Data, typename T_Value, typename T_Index>
    struct IsWriteAccessor< T_Accessor, T_Data, T_Value, T_Index, void_t<
            std::result_of_t<T_Accessor(T_Index, T_Data&, T_Value)>
        >
    >: std::true_type{};


    template<
        class T_Accessor,
        typename T_Data,
        typename T_Value = char,
        typename T_SFINAE = void
    >
    struct IsStreamAccessor: std::false_type{};

    template< class T_Accessor, typename T_Data, typename T_Value>
    struct IsStreamAccessor< T_Accessor, T_Data, T_Value, void_t<
            decltype(&T_Accessor::getDelimiters)
        >
    >: std::true_type
    {
        static_assert(
                IsReadAccessor< T_Accessor, T_Data >::value ||
                IsWriteAccessor< T_Accessor, T_Data, T_Value >::value,
                "Only Accessors should be checked with this trait");
    };

    template<
        class T_Accessor,
        typename T_Data,
        typename T_Index = types::Vec< NumDims<T_Data>::value >
    >
    struct ReadAccessorReturnType
    {
        static_assert(
                IsReadAccessor< T_Accessor, T_Data >::value,
                "Only Read-Accessors should be checked with this trait");

        using type = std::result_of_t<T_Accessor(T_Index, T_Data)>;
    };

}  // namespace traits
}  // namespace foobar
