#pragma once

#include <boost/utility.hpp>
#include "foobar/traits/NumDims.hpp"

namespace foobar {
namespace policies {

    /**
     * Provides a []-operator to get the extents in the specified dimension of the data object given in the constructor
     * Row-Major order assumed, that is last dimension varies fastest
     */
    template< typename T_Data >
    struct GetExtents: private boost::noncopyable
    {
        using Data = T_Data;

        GetExtents(const Data& data): data_(data){}

        unsigned operator[](unsigned dimIdx) const
        {
            return data_.extents[dimIdx];
        }
    protected:
        const Data& data_;
    };

    /**
     * Same as \ref GetExtents but only considers the last n dimensions
     */
    template< typename T_Data, unsigned T_numDims >
    struct GetLastNExtents: private boost::noncopyable
    {
        using Data = T_Data;
        static constexpr unsigned numDims = T_numDims;
        static constexpr unsigned origNumDims = traits::NumDims< T_Data >::value;
        static_assert(origNumDims >= numDims, "Can't get more dimensions then available");
        static constexpr unsigned numSkipped = origNumDims - numDims;

        GetLastNExtents(const Data& data): data_(data){}

        unsigned operator[](unsigned dimIdx) const
        {
            return data_[dimIdx + numSkipped];
        }
    protected:
            const GetExtents< Data > data_;
    };

}  // namespace policies
}  // namespace foobar
