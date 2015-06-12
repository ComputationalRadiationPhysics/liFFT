#pragma once

#include "foobar/c++14_types.hpp"

namespace foobar {
namespace policies {

    /**
     * Accesses a Volume (1D-3D) by using its ()-operator(x, y, z) where x is the fastest varying index
     */
    struct VolumeAccessor
    {
        template<
            class T_Index,
            class T_Data,
            typename T_SFINAE = std::enable_if_t<
                (traits::NumDims<T_Data>::value == 1)
            >*
        >
        auto
        operator()(const T_Index& idx, const T_Data& data) const
        -> decltype(data(idx[0]))
        {
            return data(idx[0]);
        }

        template<
            class T_Index,
            class T_Data,
            typename T_SFINAE = std::enable_if_t<
                (traits::NumDims<T_Data>::value == 2)
            >*
        >
        auto
        operator()(const T_Index& idx, const T_Data& data) const
        -> decltype(data(idx[1], idx[0]))
        {
            return data(idx[1], idx[0]);
        }

        template<
            class T_Index,
            class T_Data,
            typename T_SFINAE = std::enable_if_t<
                (traits::NumDims<T_Data>::value == 3)
            >*
        >
        auto
        operator()(const T_Index& idx, const T_Data& data) const
        -> decltype(data(idx[2], idx[1], idx[0]))
        {
            return data(idx[2], idx[1], idx[0]);
        }
    };

}  // namespace policies
}  // namespace foobar
