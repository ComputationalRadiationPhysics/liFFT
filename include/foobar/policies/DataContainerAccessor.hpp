#pragma once

#include <foobar/policies/flattenIdx.hpp>
#include "foobar/c++14_types.hpp"
#include "foobar/util.hpp"

namespace foobar {
namespace policies {

    /**
     * Accessor that can read and write a DataContainer (type with an array-like member named data)
     */
    struct DataContainerAccessor
    {
    public:
        template< class T_Index, class T_Data >
        auto
        operator()(T_Index&& idx, T_Data& data) const
        -> decltype(
                std::declval< typename T_Data::BaseAccessor >()(
                        0, getConstCorrect<T_Data>(data.data)
                )
           )
        {
            auto flatIdx = getFlatIdx(idx, data);
            typename T_Data::BaseAccessor acc;
            return acc(flatIdx, getConstCorrect<T_Data>(data.data));
        }

        template< class T_Index, class T_Data, typename T_Value >
        void
        operator()(T_Index&& idx, T_Data& data, T_Value&& value) const
        {
            auto flatIdx = getFlatIdx(idx, data);
            typename T_Data::BaseAccessor acc;
            acc(flatIdx, getConstCorrect<T_Data>(data.data), std::forward<T_Value>(value));
        }
    };

}  // namespace policies
}  // namespace foobar
