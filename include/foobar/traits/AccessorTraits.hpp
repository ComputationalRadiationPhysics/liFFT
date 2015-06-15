#pragma once

#include "foobar/void_t.hpp"
#include "foobar/traits/NumDims.hpp"

namespace foobar {
namespace traits {

    template<
        class T_Accessor,
        typename T_Data,
        typename T_Index = types::Vec< NumDims<T_Data>::value >,
        typename T_SFINAE = void
    >
    struct IsReadArrayAccessor: std::false_type{};

    template< class T_Accessor, typename T_Data, typename T_Index>
    struct IsReadArrayAccessor< T_Accessor, T_Data, T_Index, void_t<
            typename std::result_of<T_Accessor(T_Index, T_Data)>::type
        >
    >: std::true_type{};


    template<
        class T_Accessor,
        typename T_Data,
        typename T_SFINAE = void
    >
    struct IsReadStreamAccessor: std::false_type{};

    template< class T_Accessor, typename T_Data>
    struct IsReadStreamAccessor< T_Accessor, T_Data, void_t<
            typename std::result_of<T_Accessor(T_Data)>::type
        >
    >: std::true_type{};

    template<
        class T_Accessor,
        typename T_Data,
        typename T_Index = types::Vec< NumDims<T_Data>::value >
    >
    struct ReadAccessorReturnType
    {
        using IsArray = IsReadArrayAccessor< T_Accessor, T_Data, T_Index >;
        using IsStream = IsReadStreamAccessor< T_Accessor, T_Data, T_Index >;
        static_assert(IsArray::value || IsStream::value, "Must be either array or stream accessor");

        using type = typename std::conditional_t<
                         IsArray::value,
                         std::result_of<T_Accessor(T_Index, T_Data)>,
                         std::result_of<T_Accessor(T_Data)>
                     >::type;
    };

    template<
        class T_Accessor,
        typename T_Data,
        typename T_Value,
        typename T_Index = types::Vec< NumDims<T_Data>::value >,
        typename T_SFINAE = void
    >
    struct IsWriteArrayAccessor: std::false_type{};

    template< class T_Accessor, typename T_Data, typename T_Value, typename T_Index>
    struct IsWriteArrayAccessor< T_Accessor, T_Data, T_Value, T_Index, void_t<
            typename std::result_of<T_Accessor(T_Index, T_Data&, T_Value)>::type
        >
    >: std::true_type{};


    template<
        class T_Accessor,
        typename T_Data,
        typename T_Value,
        typename T_SFINAE = void
    >
    struct IsWriteStreamAccessor: std::false_type{};

    template< class T_Accessor, typename T_Data, typename T_Value>
    struct IsWriteStreamAccessor< T_Accessor, T_Data, T_Value, void_t<
            typename std::result_of<T_Accessor(T_Data&, T_Value)>::type
        >
    >: std::true_type{};


}  // namespace traits
}  // namespace foobar
