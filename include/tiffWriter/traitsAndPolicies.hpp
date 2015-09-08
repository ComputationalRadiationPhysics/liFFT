#pragma once

#include "tiffWriter/image.hpp"
#include "foobar/traits/NumDims.hpp"
#include "foobar/traits/IdentityAccessor.hpp"
#include "foobar/traits/IsComplex.hpp"
#include "foobar/traits/IsStrided.hpp"
#include "foobar/accessors/VolumeAccessor.hpp"
#include "foobar/policies/GetExtents.hpp"
#include "foobar/types/Vec.hpp"
#include "foobar/util.hpp"

namespace foobar {
namespace traits {

    template< tiffWriter::ImageFormat T_imgFormat, class T_Allocator >
    struct NumDims< tiffWriter::Image< T_imgFormat, T_Allocator > >: std::integral_constant<unsigned, 2>{};

    template< tiffWriter::ImageFormat T_imgFormat, class T_Allocator >
    struct IsComplex< tiffWriter::Image< T_imgFormat, T_Allocator > >: BoolConst<false>{};

    template< tiffWriter::ImageFormat T_imgFormat, class T_Allocator >
    struct IsStrided< tiffWriter::Image< T_imgFormat, T_Allocator > >: BoolConst<false>{};

    template< tiffWriter::ImageFormat T_imgFormat, class T_Allocator >
    struct IdentityAccessor< tiffWriter::Image< T_imgFormat, T_Allocator > >
    {
        using type = accessors::VolumeAccessor;
    };

}  // namespace traits

namespace policies {

    template< tiffWriter::ImageFormat T_imgFormat, class T_Allocator >
    struct GetExtentsImpl< tiffWriter::Image< T_imgFormat, T_Allocator > >
    {
        using type = tiffWriter::Image< T_imgFormat, T_Allocator >;

        GetExtentsImpl(const type& data): extents_(data.getHeight(), data.getWidth()){}

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
