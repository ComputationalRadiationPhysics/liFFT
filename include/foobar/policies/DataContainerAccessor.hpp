#pragma once

#include "foobar/policies/GetExtents.hpp"
#include "foobar/c++14_types.hpp"
#include "foobar/util.hpp"

namespace foobar {
namespace policies {

    /**
     * Accessor that can read and write a DataContainer (type with an array-like member named data)
     */
    struct DataContainerAccessor
    {
    protected:
        template< class T_Index, class T_Data >
        std::enable_if_t< std::is_integral<T_Index>::value, unsigned >
        getFlatIdx(const T_Index& idx, const T_Data& data) const
        {
            return idx;
        }

        template< class T_Index, class T_Data >
        std::enable_if_t< !std::is_integral<T_Index>::value, unsigned >
        getFlatIdx(const T_Index& idx, const T_Data& data) const
        {
            static constexpr unsigned numDims = traits::NumDims<T_Data>::value;
            GetExtents<T_Data> extents(data);
            unsigned flatIdx = idx[0];
            for(unsigned i=1; i<numDims; ++i)
                flatIdx = flatIdx*extents[i] + idx[i];
            return flatIdx;
        }
    public:

        template< class T_Index, class T_Data >
        auto
        operator()(T_Index&& idx, T_Data& data) const
        -> decltype(std::declval< typename T_Data::Accessor >()(0, data.data))
        {
            auto flatIdx = getFlatIdx(idx, data);
            typename T_Data::Accessor acc;
            return acc(flatIdx, data.data);
        }

        template< class T_Index, class T_Data, typename T_Value >
        void
        operator()(T_Index&& idx, T_Data& data, T_Value&& value) const
        {
            auto flatIdx = getFlatIdx(idx, data);
            typename T_Data::Accessor acc;
            acc(flatIdx, data.data, std::forward<T_Value>(value));
        }
    };

}  // namespace policies
}  // namespace foobar
