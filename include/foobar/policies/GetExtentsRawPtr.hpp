#pragma once

#include <boost/utility.hpp>

namespace foobar {
namespace policies {

    /**
     * Default implementation when we already have a contiguous array
     */
    template< typename T_Data, bool T_copy = false, unsigned T_numDims = traits::NumDims<T_Data>::value >
    struct GetExtentsRawPtrImpl: private boost::noncopyable
    {
        using Data = T_Data;
        static constexpr unsigned numDims = T_numDims;

        GetExtentsRawPtrImpl(const Data& data): value_(const_cast<Data&>(data).extents.data()){}

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
    template< typename T_Data, unsigned T_numDims >
    struct GetExtentsRawPtrImpl< T_Data, true, T_numDims >
    {
        using Data = T_Data;
        static constexpr unsigned numDims = T_numDims;

        GetExtentsRawPtrImpl(const Data& data){
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

    /**
     * Functor that returns a raw ptr to an unsigned int array
     * containing 1 entry per dimension with the extents in that dimensions
     */
    template< typename T_Data, unsigned T_numDims = traits::NumDims<T_Data>::value >
    struct GetExtentsRawPtr: GetExtentsRawPtrImpl< T_Data, false, T_numDims >{
        using Data = T_Data;
        using Parent = GetExtentsRawPtrImpl< T_Data, false, T_numDims >;
        static constexpr unsigned numDims = T_numDims;

        GetExtentsRawPtr(const Data& data):Parent(data){}
    };

}  // namespace policies
}  // namespace foobar
