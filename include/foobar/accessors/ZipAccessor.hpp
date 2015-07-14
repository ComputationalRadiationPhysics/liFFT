#pragma once

#include "foobar/traits/DefaultAccessor.hpp"

namespace foobar {
namespace accessors {

    /***
     * Accessor that zips together 2 datasets
     * That is an access to (idx, data) will yield Func(AccFirst(idx, dataFirst), AccSecond(idx, data))
     * Therefore you MUST make sure, that the dimensions of dataFirst are at least as big as data!
     */
    template< class T_DataFirst, class T_Func, class T_AccSecond, class T_AccFirst = traits::DefaultAccessor_t<T_DataFirst> >
    struct ZipAccessor
    {
        using AccFirst = T_AccFirst;
        using AccSecond = T_AccSecond;
        using DataFirst= T_DataFirst;
        using Func = T_Func;

        AccFirst accFirst_;
        AccSecond accSecond_;
        DataFirst& dataFirst_;
        Func func_;

        ZipAccessor(DataFirst& dataFirst, const Func& func = Func(), const AccSecond& accSecond = AccSecond(), const AccFirst& accFirst = AccFirst()):
            dataFirst_(dataFirst), func_(func), accSecond_(accSecond), accFirst_(accFirst)
        {}

        template< typename T_Idx, class T_Data >
        auto
        operator()(T_Idx& idx, T_Data& data)
        -> decltype( func_( accFirst_(idx, dataFirst_), accSecond_(idx, data) ) )
        {
            return func_( accFirst_(idx, dataFirst_), accSecond_(idx, data) );
        }
    };

    template< class T_DataFirst, class T_Func, class T_AccSecond, class T_AccFirst = traits::DefaultAccessor_t<T_DataFirst> >
    ZipAccessor< T_DataFirst, T_Func, T_AccSecond, T_AccFirst>
    makeZipAccessor(T_DataFirst& dataFirst, const T_Func& func, const T_AccSecond& accSecond, const T_AccFirst& accFirst = T_AccFirst())
    {
        return ZipAccessor< T_DataFirst, T_Func, T_AccSecond, T_AccFirst>(dataFirst, func, accSecond, accFirst);
    }


}  // namespace accessors
}  // namespace foobar
