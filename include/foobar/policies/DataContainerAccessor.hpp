#pragma once

namespace foobar {
namespace policies {

    /**
     * Accessor that can read and write a DataContainer (type with an array-like member named data)
     */
    struct DataContainerAccessor
    {
        template< class T_Index, class T_Data >
        unsigned
        getFlatIdx(const T_Index& idx, const T_Data& data) const
        {
            static constexpr unsigned numDims = traits::NumDims<T_Data>::value;
            GetExtents<T_Data> extents(data);
            unsigned flatIdx = idx[0];
            for(unsigned i=1; i<numDims; ++i)
                flatIdx = flatIdx*extents[i] + idx[i];
            return flatIdx;
        }

        template< class T_Index, class T_Data >
        auto
        operator()(const T_Index& idx, const T_Data& data) const
        -> typename std::remove_pointer< decltype(data.data) >::type
        {
            using DataType = decltype(T_Data::data);
            using ReturnType = typename std::remove_pointer< DataType >::type;
            static_assert(!std::is_same< DataType, ReturnType >::value, "This only works for pointer data types");

            auto flatIdx = getFlatIdx(idx, data);
            return data.data[flatIdx];
        }

        template< class T_Index, class T_Data >
        void
        operator()(const T_Index& idx, T_Data& data, typename std::remove_pointer< decltype(data.data) >::type&& value)
        {
            using DataType = decltype(data.data);
            using ReturnType = typename std::remove_pointer< DataType >::type;
            static_assert(!std::is_same< DataType, ReturnType >::value, "This only works for pointer data types");

            auto flatIdx = getFlatIdx(idx, data);
            data.data[flatIdx] = value;
        }
    };

}  // namespace policies
}  // namespace foobar
