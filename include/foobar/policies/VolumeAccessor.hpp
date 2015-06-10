#pragma once

namespace foobar {
namespace policies {

    struct VolumeAccessor
    {
        template< class T_Index, class T_Data >
        auto
        operator()(const T_Index& idx, const T_Data& data) const
        -> decltype(data(idx[1], idx[0]))
        {
            return data(idx[1], idx[0]);
        }
    };

}  // namespace policies
}  // namespace foobar
