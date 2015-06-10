#pragma once

#include <boost/utility.hpp>

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

}  // namespace policies
}  // namespace foobar
