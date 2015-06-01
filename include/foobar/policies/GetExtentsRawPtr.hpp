#pragma once

#include <boost/utility.hpp>

namespace foobar {
namespace policies {

    /**
     * Functor that returns a raw ptr to an unsigned int array
     * containing 1 entry per dimension with the extents in that dimensions
     */
    template< typename T_Data, bool T_copy = false >
    struct GetExtentsRawPtr: private boost::noncopyable
    {
        using Data = T_Data;

        GetExtentsRawPtr(const Data& data): value_(const_cast<Data&>(data).extents.data()){}

        unsigned* operator()()
        {
            return value_;
        }
    private:
        unsigned* value_;

    };

    /**
     * Partial specialization when an internal contiguous array has to be allocated
     */
    template< typename T_Data >
    struct GetExtentsRawPtr< T_Data, true >
    {
        using Data = T_Data;
        static constexpr unsigned numDims = traits::NumDims<T_Data>::value;

        GetExtentsRawPtr(const Data& data){
            GetExtents<T_Data> extents(data);
            for(unsigned i=0; i<numDims; ++i)
                extents_[i] = extents[i];
        }

        unsigned* operator()()
        {
            return extents_.data();
        }
    private:
        std::array< unsigned, numDims > extents_;
    };

}  // namespace policies
}  // namespace foobar
