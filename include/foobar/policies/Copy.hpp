#pragma once

#include "foobar/traits/NumDims.hpp"
#include "foobar/traits/AccessorTraits.hpp"
#include "foobar/policies/GetExtents.hpp"
#include "foobar/types/Vec.hpp"

namespace foobar {
namespace policies {
    namespace detail {

        template<bool T_lastDim>
        struct CopyImpl
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
                    if(idx[T_curDim] > 0){
                        accSrc.readDelimiter(src, T_curDim);
                        accDst.writeDelimiter(dst, T_curDim);
                    }
                    accDst.write(idx, dst, accSrc.read(idx, src));
                }
            }
        };

        template<>
        struct CopyImpl<false>
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
                    if(idx[T_curDim] > 0){
                        accSrc.readDelimiter(src, T_curDim);
                        accDst.writeDelimiter(dst, T_curDim);
                    }
                    CopyImpl< (T_curDim+2 == T_lastDim) >::
                            template loop< T_curDim+1, T_lastDim >(idx, extents, src, std::forward<T_SrcAccessor>(accSrc), dst, std::forward<T_DstAccessor>(accDst));
                }
            }
        };

        template< class T_BaseAccessor >
        struct ReadAccessorWrapper
        {
            using BaseAccessor = T_BaseAccessor;

            BaseAccessor acc_;

            ReadAccessorWrapper(){}
            ReadAccessorWrapper(const BaseAccessor& acc): acc_(acc){}
            ReadAccessorWrapper(BaseAccessor&& acc): acc_(std::move(acc)){}

            template< typename T_Data >
            std::enable_if_t< traits::IsStreamAccessor< BaseAccessor, T_Data >::value >
            readDelimiter(T_Data& data, unsigned dim)
            {
                acc_.skipDelimiter(data, dim);
            }

            template< typename T_Data >
            std::enable_if_t< !traits::IsStreamAccessor< BaseAccessor, T_Data >::value >
            readDelimiter(T_Data& data, unsigned dim){}

            template<
                typename T_Index,
                typename T_Data
            >
            auto
            read(const T_Index& idx, T_Data& data)
            -> decltype(acc_(idx, data))
            {
                return acc_(idx, data);
            }
        };

        template< class T_BaseAccessor >
        struct WriteAccessorWrapper
        {
            using BaseAccessor = T_BaseAccessor;

            BaseAccessor acc_;

            WriteAccessorWrapper(){}
            WriteAccessorWrapper(const BaseAccessor& acc): acc_(acc){}
            WriteAccessorWrapper(BaseAccessor&& acc): acc_(std::move(acc)){}

            template< typename T_Data >
            std::enable_if_t< traits::IsStreamAccessor< BaseAccessor, T_Data, char >::value >
            writeDelimiter(T_Data& data, unsigned dim)
            {
                types::Vec<1> dummyIdx;
                acc_(dummyIdx, data, acc_.getDelimiters()[dim]);
            }

            template< typename T_Data >
            std::enable_if_t< !traits::IsStreamAccessor< BaseAccessor, T_Data, char >::value >
            writeDelimiter(T_Data& data, unsigned dim){}

            template<
                typename T_Index,
                typename T_Data,
                typename T_Value
            >
            void
            write(const T_Index& idx, T_Data& data, T_Value&& value)
            {
                acc_(idx, data, std::forward<T_Value>(value));
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
     * @param T_DstAccessor Accessor used to set an element in dst  : operator(idx, dst, value)
     * For Stream accessors:
     *          getDelimiters() -> array like type with delimiters for each dimension
     * For Stream-Read accessors:
     *          skipDelimiter(data, curDim) -> Skips the delimiter of this dimension in the stream
     */
    template< class T_SrcAccessor, class T_DstAccessor >
    struct Copy
    {
    private:
        detail::ReadAccessorWrapper< T_SrcAccessor > accSrc_;
        detail::WriteAccessorWrapper< T_DstAccessor > accDst_;

        template< class T_Accessor, unsigned T_numDims, bool T_isStreamAcc = false >
        struct DelimiterDimOk: std::true_type{};

        template< class T_Accessor, unsigned T_numDims >
        struct DelimiterDimOk< T_Accessor, T_numDims, true >
        {
            static constexpr bool value = T_numDims <= traits::NumDims<decltype(accDst_.acc_.getDelimiters())>::value;
        };
    public:
        Copy(){}
        Copy(T_SrcAccessor accSrc, T_DstAccessor accDst): accSrc_(accSrc), accDst_(accDst){}

        template< class T_Src, class T_Dst >
        void
        operator()(const T_Src& src, T_Dst& dst)
        {
            static constexpr unsigned numDims    = traits::NumDims<T_Src>::value;
            static constexpr unsigned numDimsDst = traits::NumDims<T_Dst>::value;
            static_assert(numDims == numDimsDst, "Dimensions must match");

            static_assert(DelimiterDimOk<T_SrcAccessor, numDims, traits::IsStreamAccessor<T_SrcAccessor, T_Src>::value>::value,
                    "Source accessor does not provide enough delimiters");
            static_assert(DelimiterDimOk<T_DstAccessor, numDims, traits::IsStreamAccessor<T_DstAccessor, T_Dst>::value>::value,
                    "Destination accessor does not provide enough delimiters");

            using ExtentsVec = types::Vec<numDims>;
            ExtentsVec idx;
            detail::CopyImpl< (numDims == 1) >::
                    template loop< 0, numDims >(idx, GetExtents<T_Src>(src), src, accSrc_, dst, accDst_);
        }
    };

}  // namespace policies
}  // namespace foobar
