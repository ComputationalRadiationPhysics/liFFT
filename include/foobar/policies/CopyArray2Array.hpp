#pragma once

#include "foobar/traits/NumDims.hpp"
#include "foobar/policies/LoopNDims.hpp"
#include "foobar/policies/GetExtents.hpp"
#include "foobar/types/Vec.hpp"

namespace foobar {
namespace policies {

    /**
     * Policy that copies the contents of an array(-like) type to another
     */
    template< class T_SrcAccessor >
    struct CopyArray2Array
    {
        T_SrcAccessor accSrc_;

        template< class T_Src, class T_Dst, class T_DstAccessor >
        void
        operator()(const T_Src& src, T_Dst& dst, T_DstAccessor&& accDst)
        {
            static constexpr unsigned numDims = traits::NumDims<T_Src>::value;
            static_assert(numDims == traits::NumDims<T_Dst>::value, "Dimensions must match");
            using ExtentsVec = types::Vec<numDims>;
            auto func = [&](const ExtentsVec& idx)
            {
                accDst(idx, dst, accSrc_(idx, src));
            };
            policies::LoopNDims<numDims>::template loop(ExtentsVec(), GetExtents<T_Src>(src), func);
        }
    };

}  // namespace policies
}  // namespace foobar
