#pragma once

namespace foobar {
namespace policies {

    /**
     * Accessor that transforms all elements on access using the specified functor
     */
    template< class T_BaseAccessor, class T_Func >
    struct TransformAccessor
    {
    private:
        using BaseAccessor = T_BaseAccessor;
        using Func = T_Func;

        BaseAccessor acc_;
        Func func_;
    public:

        template< class T, class U >
        TransformAccessor(T&& baseAccessor, U&& func): acc_(std::forward<T>(baseAccessor)), func_(std::forward<U>(func)){}

        template< class T >
        TransformAccessor(T&& baseAccessor): acc_(std::forward<T>(baseAccessor)){}

        TransformAccessor(){}

        template< class T_Index, class T_Data >
        auto
        operator()(const T_Index& idx, const T_Data& data) const
        -> decltype(func_(acc_(idx, data)))
        {
            return func_(acc_(idx, data));
        }
    };

    template< class T_BaseAccessor, class T_Func >
    TransformAccessor< T_BaseAccessor, T_Func >
    makeTransformAccessor(T_BaseAccessor&& acc, T_Func&& func)
    {
        return TransformAccessor< T_BaseAccessor, T_Func >(std::forward<T_BaseAccessor>(acc), std::forward<T_Func>(func));
    }

}  // namespace policies
}  // namespace foobar
