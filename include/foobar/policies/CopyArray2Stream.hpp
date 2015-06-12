#pragma once

#include "foobar/traits/NumDims.hpp"
#include "foobar/policies/LoopNDims.hpp"
#include "foobar/policies/GetExtents.hpp"
#include "foobar/types/Vec.hpp"

namespace foobar {
namespace policies {

    namespace detail {

        template<bool T_lastDim>
        struct CopyArray2StreamImpl
        {
            template<
                unsigned T_curDim,
                unsigned T_endDim,
                class T_Index,
                class T_Extents,
                class T_Src,
                class T_SrcAccessor,
                class T_Dst,
                class T_DstAccessor
            >
            static void
            loop(T_Index& idx, const T_Extents& extents, const T_Src& src, T_SrcAccessor&& accSrc, T_Dst& dst, T_DstAccessor&& accDst)
            {
                for(idx[T_curDim]=0; idx[T_curDim]<extents[T_curDim]; ++idx[T_curDim])
                {
                    if(idx[T_curDim] > 0)
                        accDst(dst, accDst.getDelimiters()[T_curDim]);
                    accDst(dst, accSrc(idx, src));
                }
            }
        };

        template<>
        struct CopyArray2StreamImpl<false>
        {
            template<
                unsigned T_curDim,
                unsigned T_lastDim,
                class T_Index,
                class T_Extents,
                class T_Src,
                class T_SrcAccessor,
                class T_Dst,
                class T_DstAccessor
            >
            static void
            loop(T_Index& idx, const T_Extents& extents, const T_Src& src, T_SrcAccessor&& accSrc, T_Dst& dst, T_DstAccessor&& accDst)
            {
                for(idx[T_curDim]=0; idx[T_curDim]<extents[T_curDim]; ++idx[T_curDim])
                {
                    if(idx[T_curDim] > 0)
                        accDst(dst, accDst.getDelimiters()[T_curDim]);
                    CopyArray2StreamImpl< (T_curDim+2 == T_lastDim) >::
                            template loop< T_curDim+1, T_lastDim >(idx, extents, src, std::forward<T_SrcAccessor>(accSrc), dst, std::forward<T_DstAccessor>(accDst));
                }
            }
        };

    }  // namespace detail

    /**
     * Policy that copies the contents of an array(-like) type to a stream(-like) type
     * That is something that can't be accessed with indices
     *
     * Provides a ()-operator(source, destination)
     *
     * @param T_SrcAccessor Accessor used to get an element from src: <type> operator(idx, src)
     * @param T_DstAccessor Accessor used to set an element in dst  :
     *          operator(dst, value)
     *          getDelimiters() -> array like type with delimiters for each dimension
     */
    template< class T_SrcAccessor, class T_DstAccessor >
    struct CopyArray2Stream
    {
        T_SrcAccessor accSrc_;
        T_DstAccessor accDst_;

        template< class T_Src, class T_Dst >
        void
        operator()(const T_Src& src, T_Dst& dst)
        {
            static constexpr unsigned numDims    = traits::NumDims<T_Src>::value;
            static constexpr unsigned numDimsDst = traits::NumDims<T_Dst>::value;
            static_assert(numDims == 2, "Dimensions must match");
            static_assert(2 == numDimsDst, "Dimensions must match");
            static_assert(numDims == numDimsDst, "Dimensions must match");

            static_assert(numDims <= traits::NumDims<decltype(accDst_.getDelimiters())>::value, "Accessor does not provide enough delimiters");
            using ExtentsVec = types::Vec<numDims>;
            ExtentsVec idx;
            detail::CopyArray2StreamImpl< (numDims == 1) >::
                    template loop< 0, numDims >(idx, GetExtents<T_Src>(src), src, accSrc_, dst, accDst_);
        }
    };

}  // namespace policies
}  // namespace foobar
