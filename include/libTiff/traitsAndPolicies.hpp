#pragma once

#include "libTiff/libTiff.hpp"
#include "foobar/traits/NumDims.hpp"
#include "foobar/policies/GetExtents.hpp"
#include "foobar/types/Vec.hpp"

namespace foobar {
namespace traits {

    template<class T_Allocator>
    struct NumDims< libTiff::TiffImage<T_Allocator> >: std::integral_constant<unsigned, 2>{};

}  // namespace traits

namespace policies {

    template<class T_Allocator>
    struct GetExtents< libTiff::TiffImage<T_Allocator> >
    {
        using type = libTiff::TiffImage<T_Allocator>;

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
