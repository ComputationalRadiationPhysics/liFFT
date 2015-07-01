#pragma once

#include "foobar/traits/NumDims.hpp"
#include "foobar/policies/GetExtents.hpp"
#include "foobar/types/Vec.hpp"
#include "foobar/c++14_types.hpp"

namespace foobar {
namespace policies {

    namespace detail {

        template<bool T_lastDim>
        struct LoopImpl
        {
            template<
                unsigned T_curDim,
                unsigned T_endDim,
                class T_Index,
                class T_Extents,
                class T_Src,
                class T_SrcAccessor,
                class T_Dst,
                class T_DstAccessor,
                class T_Handler
            >
            static void
            loop(T_Index& idx, const T_Extents& extents, T_Src& src, T_SrcAccessor&& accSrc, T_Dst& dst, T_DstAccessor&& accDst, T_Handler& handler)
            {
                for(idx[T_curDim]=0; idx[T_curDim]<extents[T_curDim]; ++idx[T_curDim])
                {
                    handler.template handleInnerLoop<T_curDim>(idx, src, accSrc, dst, accDst);
                }
            }
        };

        template<>
        struct LoopImpl<false>
        {
            template<
                unsigned T_curDim,
                unsigned T_endDim,
                class T_Index,
                class T_Extents,
                class T_Src,
                class T_SrcAccessor,
                class T_Dst,
                class T_DstAccessor,
                class T_Handler
            >
            static void
            loop(T_Index& idx, const T_Extents& extents, T_Src& src, T_SrcAccessor&& accSrc, T_Dst& dst, T_DstAccessor&& accDst, T_Handler& handler)
            {
                for(idx[T_curDim]=0; idx[T_curDim]<extents[T_curDim]; ++idx[T_curDim])
                {
                    handler.template handleLoopPre< T_curDim, T_endDim >(idx, src, accSrc, dst, accDst);
                    LoopImpl< (T_curDim+2 == T_endDim) >::
                            template loop< T_curDim+1, T_endDim >(idx, extents, src, std::forward<T_SrcAccessor>(accSrc), dst, std::forward<T_DstAccessor>(accDst), handler);
                    handler.template handleLoopPost< T_curDim, T_endDim >(idx, src, accSrc, dst, accDst);
                }
            }
        };

    }  // namespace detail

    /**
     * Defines a loop over all dimensions of src
     * Expects a handler that is then called for each iteration. Specifically:
     * Before any inner loop:  handler.handleLoopPre< curDim, endDim >(idx, src, accSrc, dst, accDst)
     * After any inner loop:   handler.handleLoopPost< curDim, endDim >(idx, src, accSrc, dst, accDst)
     * In the most inner loop: handler.handleInnerLoop< curDim >(idx, src, accSrc, dst, accDst)
     */
    template<
        class T_Src,
        class T_Dst,
        class T_SrcAccessor = const foobar::traits::DefaultAccessor_t<T_Src>,
        class T_DstAccessor = const foobar::traits::DefaultAccessor_t<T_Dst>
    >
    struct Loop
    {
    public:
        template< class T_Handler >
        static void
        execute(T_Src& src, T_Dst& dst, T_Handler&& handler, T_SrcAccessor& accSrc = T_SrcAccessor(), T_DstAccessor& accDst = T_DstAccessor())
        {
            static constexpr unsigned numDims    = traits::NumDims<std::remove_const_t<T_Src>>::value;
            static constexpr unsigned numDimsDst = traits::NumDims<std::remove_const_t<T_Dst>>::value;
            static_assert(numDims == numDimsDst, "Dimensions must match");

            using ExtentsVec = types::Vec<numDims>;
            ExtentsVec idx;
            detail::LoopImpl< (numDims == 1) >::
                    template loop< 0, numDims >(idx, GetExtents<T_Src>(src), src, accSrc, dst, accDst, handler);
        }
    };

    template<
        class T_Src,
        class T_Dst,
        class T_Handler,
        class T_SrcAccessor = const foobar::traits::DefaultAccessor_t<T_Src>,
        class T_DstAccessor = const foobar::traits::DefaultAccessor_t<T_Dst>
    >
    void
    loop(T_Src& src, T_Dst& dst, T_Handler&& handler, T_SrcAccessor& accSrc = T_SrcAccessor(), T_DstAccessor& accDst = T_DstAccessor())
    {
        Loop< T_Src, T_Dst, T_SrcAccessor, T_DstAccessor >::execute(src, dst, handler, accSrc, accDst);
    }

}  // namespace policies
}  // namespace foobar
