#pragma once

#include "foobar/policies/GetExtents.hpp"
#include "foobar/policies/GetStrides.hpp"
#include "foobar/traits/NumDims.hpp"
#include "foobar/traits/IsStrided.hpp"

namespace foobar {
namespace policies {

    /**
     * Makes an index "flat", that is: vector indices are converted to an unsigned
     */
    template< class T_Data, bool T_IsStrided = traits::IsStrided<T_Data> >
    struct FlattenIdx
    {
        template< class T_Index >
        std::enable_if_t< std::is_integral<T_Index>::value, unsigned >
        operator()(T_Index&& idx, const T_Data& data) const
        {
            return idx;
        }

        template< class T_Index >
        std::enable_if_t< !std::is_integral<T_Index>::value, unsigned >
        operator()(T_Index&& idx, const T_Data& data) const
        {
            static constexpr unsigned numDims = traits::NumDims<T_Data>::value;
            GetExtents<T_Data> extents(data);
            unsigned flatIdx = idx[0];
            for(unsigned i=1; i<numDims; ++i)
                flatIdx = flatIdx*extents[i] + idx[i];
            return flatIdx;
        }
    };

    template< class T_Data >
    struct FlattenIdx< T_Data, true>
    {
        template< class T_Index >
        std::enable_if_t< std::is_integral<T_Index>::value, unsigned >
        operator()(T_Index&& idx, const T_Data& data) const
        {
            return idx;
        }

        template< class T_Index >
        std::enable_if_t< !std::is_integral<T_Index>::value, unsigned >
        operator()(T_Index&& idx, const T_Data& data) const
        {
            static constexpr unsigned numDims = traits::NumDims<T_Data>::value;
            GetStrides<T_Data> strides(data);
            unsigned flatIdx = 0;
            for(unsigned i=0; i<numDims; ++i)
                flatIdx += idx[i] * strides[i];
            return flatIdx;
        }
    };

    template< class T_Index, class T_Data >
    unsigned
    flattenIdx(T_Index&& idx, const T_Data& data)
    {
        return FlattenIdx<T_Data>()(std::forward<T_Index>(idx), data);
    }

}  // namespace policies
}  // namespace foobar
