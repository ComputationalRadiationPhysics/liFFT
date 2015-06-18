#pragma once

namespace foobar {
namespace policies {

    /**
     * Accessor that access the data directly via the index
     *
     * \tparam T_isFunctor When false, the array is accessed via []-operator, else via ()-operator
     */
    template< bool T_isFunctor = false >
    struct ArrayAccessor
    {
    public:
        template< class T_Index, class T_Data >
        auto
        operator()(const T_Index& idx, T_Data& data) const
        -> decltype(data[idx])
        {
            return data[idx];
        }

        template< class T_Index, class T_Data, typename T_Value >
        void
        operator()(const T_Index& idx, T_Data& data, T_Value&& value) const
        {
            data[idx] = std::forward<T_Value>(value);
        }
    };

    template<>
    struct ArrayAccessor< true >
    {
    public:
        template< class T_Index, class T_Data >
        auto
        operator()(const T_Index& idx, T_Data& data) const
        -> decltype(data(idx))
        {
            return data(idx);
        }

        template< class T_Index, class T_Data, typename T_Value >
        void
        operator()(const T_Index& idx, T_Data& data, T_Value&& value) const
        {
            data(idx) = std::forward<T_Value>(value);
        }
    };

}  // namespace policies
}  // namespace foobar
