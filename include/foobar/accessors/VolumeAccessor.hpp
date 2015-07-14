#pragma once

#include "foobar/c++14_types.hpp"

namespace foobar {
namespace accessors {

    /**
     * Accesses a Volume (1D-3D) by using its ()-operator(x, y, z) where x is the fastest varying index
     */
    struct VolumeAccessor
    {
        template<
            class T_Index,
            class T_Data,
            typename T_SFINAE = std::enable_if_t<
                (traits::NumDims<std::decay_t<T_Data>>::value == 1)
            >*
        >
        auto
        operator()(const T_Index& idx, T_Data& data) const
        -> decltype(data(idx[0]))
        {
            return data(idx[0]);
        }

        template<
            class T_Index,
            class T_Data,
            typename T_SFINAE = std::enable_if_t<
                (traits::NumDims<std::decay_t<T_Data>>::value == 2)
            >*
        >
        auto
        operator()(const T_Index& idx, T_Data& data) const
        -> decltype(data(idx[1], idx[0]))
        {
            return data(idx[1], idx[0]);
        }

        template<
            class T_Index,
            class T_Data,
            typename T_SFINAE = std::enable_if_t<
                (traits::NumDims<std::decay_t<T_Data>>::value == 3)
            >*
        >
        auto
        operator()(const T_Index& idx, T_Data& data) const
        -> decltype(data(idx[2], idx[1], idx[0]))
        {
            return data(idx[2], idx[1], idx[0]);
        }
    };

}  // namespace accessors
}  // namespace foobar
