#pragma once

#include "foobar/traits/NumDims.hpp"
#include "foobar/policies/LoopNDims.hpp"
#include "foobar/policies/GetExtents.hpp"
#include "foobar/types/Vec.hpp"

namespace foobar {
namespace policies {
namespace detail {

    /**
     * Policy that copies the contents of an array(-like) type to another
     * Provides a ()-operator(source, destination)
     *
     * @param T_SrcAccessor Accessor used to get an element from src: <type> operator(idx, src)
     * @param T_DstAccessor Accessor used to set an element in dst  : operator(idx, dst, value)
     */
    template< class T_SrcAccessor, class T_DstAccessor >
    struct CopyArray2Array
    {
        T_SrcAccessor& accSrc_;
        T_DstAccessor& accDst_;

        CopyArray2Array(T_SrcAccessor& accSrc, T_DstAccessor& accDst): accSrc_(accSrc), accDst_(accDst){}

        template< class T_Src, class T_Dst >
        void
        operator()(const T_Src& src, T_Dst& dst)
        {
            static constexpr unsigned numDims = traits::NumDims<T_Src>::value;
            static_assert(numDims == traits::NumDims<T_Dst>::value, "Dimensions must match");
            using ExtentsVec = types::Vec<numDims>;
            auto func = [&](const ExtentsVec& idx)
            {
                accDst_(idx, dst, accSrc_(idx, src));
            };
            policies::LoopNDims<numDims>::template loop(ExtentsVec(), GetExtents<T_Src>(src), func);
        }
    };

}  // namespace detail
}  // namespace policies
}  // namespace foobar
