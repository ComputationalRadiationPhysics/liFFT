#pragma once

#include "foobar/util.hpp"
#include "foobar/c++14_types.hpp"

namespace foobar {
namespace accessors {

    /**
     * Accessor that uses a reinterpret cast to convert the type returned from accessor to target type
     */
    template< class T_BaseAccessor, typename T_Target >
    struct ConvertAccessor
    {
    private:
        using BaseAccessor = T_BaseAccessor;
        using Target = T_Target;

        BaseAccessor m_acc;

        template< class T_Index, class T_Data >
        struct GetCurTarget
        {
            static constexpr bool isConst = std::is_const<T_Data>::value;
            using AccRetType = std::result_of_t< BaseAccessor(const T_Index&, T_Data&) >;
            using ConstCorrectTarget = std::conditional_t<
                                            std::is_const<AccRetType>::value,
                                            foobar::AddConstVal_t<Target>,
                                            Target
                                       >;
            using type = std::conditional_t<
                            std::is_reference<AccRetType>::value,
                            typename std::add_lvalue_reference<ConstCorrectTarget>::type,
                            ConstCorrectTarget
                         >;
        };
    public:

        ConvertAccessor(){}
        template< class T >
        explicit ConvertAccessor(T&& baseAccessor): m_acc(std::forward<T>(baseAccessor)){}

        template< class T_Index, class T_Data >
        auto
        operator()(const T_Index& idx, T_Data& data) const
        -> typename GetCurTarget< T_Index, T_Data >::type
        {
            using CurTarget = typename GetCurTarget< T_Index, T_Data >::type;
            return reinterpret_cast<CurTarget>(m_acc(idx, data));
        }
    };

} // namespace accessors
} // namespace foobar
