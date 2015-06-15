#pragma once

#include "foobar/traits/AccessorTraits.hpp"
#include "foobar/policies/CopyArray2Array.hpp"
#include "foobar/policies/CopyArray2Stream.hpp"

namespace foobar {
namespace policies {

    /**
     * Policy that copies the contents of an array(-like) type to another
     * Provides a ()-operator(source, destination)
     *
     * @param T_SrcAccessor Accessor used to get an element from src: <type> operator([idx,] src)
     * @param T_DstAccessor Accessor used to set an element in dst  : operator([idx,] dst, value)
     */
    template< class T_SrcAccessor, class T_DstAccessor >
    struct Copy
    {
        T_SrcAccessor accSrc_;
        T_DstAccessor accDst_;

        template< class T_Src, class T_Dst >
        void
        operator()(const T_Src& src, T_Dst& dst)
        {
            static constexpr bool isReadArray = traits::IsReadArrayAccessor< T_SrcAccessor, T_Src >::value;
            static constexpr bool isReadStream = traits::IsReadStreamAccessor< T_SrcAccessor, T_Src >::value;
            static_assert(isReadArray || isReadStream, "Need either an array or stream accessor");
            static_assert(isReadArray, "Copy from stream no implemented");
            using ReadType = typename traits::ReadAccessorReturnType< T_SrcAccessor, T_Src >::type;

            static constexpr bool isWriteArray = traits::IsWriteArrayAccessor< T_DstAccessor, T_Dst, ReadType >::value;
            static constexpr bool isWriteStream = traits::IsWriteStreamAccessor< T_DstAccessor, T_Dst, ReadType >::value;
            static_assert(isWriteArray || isWriteStream, "Need either an array or stream accessor");

            using CopyImpl = typename std::conditional<
                    isWriteStream,
                    detail::CopyArray2Stream< T_SrcAccessor, T_DstAccessor >,
                    detail::CopyArray2Array< T_SrcAccessor, T_DstAccessor >
                    >::type;
            CopyImpl(accSrc_, accDst_)(src, dst);
        }
    };

}  // namespace policies
}  // namespace foobar
