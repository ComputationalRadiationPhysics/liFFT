#pragma once

#include <boost/utility.hpp>
#include "foobar/traits/NumDims.hpp"
#include "foobar/policies/GetExtents.hpp"

namespace foobar {
namespace policies {

    /**
     * Default implementation when we already have a contiguous array
     */
    template< typename T_Data, bool T_copy = true, unsigned T_numDims = traits::NumDims<T_Data>::value >
    struct GetExtentsRawPtrImpl: private boost::noncopyable
    {
        using Data = T_Data;
        static constexpr unsigned numDims = T_numDims;

        GetExtentsRawPtrImpl(Data& data){
            GetLastNExtents< Data, numDims > extents(data);
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

    /**
     * Partial specialization when an internal contiguous array has to be allocated
     */
    template< typename T_Data, unsigned T_numDims >
    struct GetExtentsRawPtrImpl< T_Data, false, T_numDims >
    {
        using Data = T_Data;
        static constexpr unsigned numDims = T_numDims;
        static constexpr unsigned origNumDims = traits::NumDims<T_Data>::value;
        static_assert(numDims <= origNumDims, "Type does not have that many dimensions");
        static constexpr unsigned numSkipped = origNumDims - numDims;

        GetExtentsRawPtrImpl(Data& data): value_(data.extents.data()){}

        unsigned* operator()()
        {
            return value_ + numSkipped;
        }
    private:
        unsigned* value_;
    };

    /**
     * Functor that returns a raw ptr to an unsigned int array
     * containing 1 entry per dimension with the extents in that dimensions
     * If a custom numDims value is specified only the last n dimensions are considered
     */
    template< typename T_Data, unsigned T_numDims = traits::NumDims<T_Data>::value >
    struct GetExtentsRawPtr: GetExtentsRawPtrImpl< T_Data, true, T_numDims >{
        using Data = T_Data;
        using Parent = GetExtentsRawPtrImpl< T_Data, true, T_numDims >;
        static constexpr unsigned numDims = T_numDims;

        using Parent::Parent;
    };

}  // namespace policies
}  // namespace foobar
