#pragma once

#include "foobar/policies/ArrayAccessor.hpp"

namespace foobar {
namespace policies {

    /**
     * Accessor that transposes all accesses such that the 2nd half comes before the first
     */
    template< class T_BaseAccessor = ArrayAccessor<true> >
    struct TransposeAccessor
    {
    private:
        using BaseAccessor = T_BaseAccessor;
        BaseAccessor acc_;

        template< class T_Index, class T_Data >
        void
        transposeIdx(const T_Index& idxIn, T_Index& idxOut, const T_Data& data) const
        {
            static constexpr unsigned numDims = traits::NumDims< T_Data >::value;
            GetExtents< T_Data > extents(data);
            for(unsigned i=0; i<numDims; ++i)
            {
                unsigned ext = extents[i];
                unsigned idx = idxIn[i];
                idxOut[i] = (idx >= ext/2) ? idx - ext/2 : idx + ext/2;
            }
        }
    public:

        template< typename T >
        TransposeAccessor(T&& baseAccessor): acc_(baseAccessor){}

        TransposeAccessor(){}

        template< class T_Index, class T_Data >
        auto
        operator()(const T_Index& idx, const T_Data& data) const
        -> decltype(acc_(idx, data))
        {
            T_Index transposedIdx;
            transposeIdx(idx, transposedIdx, data);
            return acc_(transposedIdx, data);
        }

        template< class T_Index, class T_Data, typename T_Value >
        void
        operator()(const T_Index& idx, T_Data& data, T_Value&& value)
        {
            T_Index transposedIdx;
            transposeIdx(idx, transposedIdx, data);
            acc_(transposedIdx, data, std::forward<T_Value>(value));
        }
    };

}  // namespace policies
}  // namespace foobar
