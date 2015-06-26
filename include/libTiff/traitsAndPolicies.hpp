#pragma once

#include "libTiff/libTiff.hpp"
#include "foobar/traits/NumDims.hpp"
#include "foobar/traits/DefaultAccessor.hpp"
#include "foobar/policies/VolumeAccessor.hpp"
#include "foobar/policies/GetExtents.hpp"
#include "foobar/types/Vec.hpp"

namespace foobar {
namespace traits {

    template< libTiff::ImageFormat T_imgFormat, class T_Allocator >
    struct NumDims< libTiff::Image< T_imgFormat, T_Allocator > >: std::integral_constant<unsigned, 2>{};

    template< libTiff::ImageFormat T_imgFormat, class T_Allocator >
    struct DefaultAccessor< libTiff::Image< T_imgFormat, T_Allocator > >
    {
        using type = policies::VolumeAccessor;
    };

}  // namespace traits

namespace policies {

    template< libTiff::ImageFormat T_imgFormat, class T_Allocator >
    struct GetExtents< libTiff::Image< T_imgFormat, T_Allocator > >
    {
        using type = libTiff::Image< T_imgFormat, T_Allocator >;

        GetExtents(const type& data): extents_(data.getHeight(), data.getWidth()){}

        unsigned
        operator[](unsigned dim) const
        {
            return extents_[dim];
        }
    private:
        const types::Vec<2> extents_;
    };

}  // namespace policies
}  // namespace foobar
