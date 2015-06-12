#pragma once

#include <boost/utility.hpp>
#include "foobar/traits/NumDims.hpp"
#include "foobar/policies/GetExtents.hpp"
#include "foobar/c++14_types.hpp"

namespace foobar {
namespace policies {

    /**
     * Default implementation when an internal contiguous array has to be allocated
     */
    template< typename T_Data, bool T_copy = true >
    struct GetExtentsRawPtrImpl: private boost::noncopyable
    {
        using Data = T_Data;
        static constexpr unsigned numDims = traits::NumDims<T_Data>::value;

        GetExtentsRawPtrImpl(Data& data){
            GetExtents< Data > extents(data);
            for(unsigned i=0; i<numDims; ++i)
                extents_[i] = extents[i];
        }

        const unsigned* operator()() const
        {
            return extents_.data();
        }
    private:
        std::array< unsigned, numDims > extents_;
    };

    /**
     * Partial specialization when we already have a contiguous array
     */
    template< typename T_Data >
    struct GetExtentsRawPtrImpl< T_Data, false >
    {
        using Data = T_Data;

        GetExtentsRawPtrImpl(Data& data): value_(data.extents.data()){}

        const unsigned* operator()() const
        {
            return value_;
        }
    private:
        unsigned* value_;
    };

    /**
     * Functor that returns a raw ptr to an unsigned int array
     * containing 1 entry per dimension with the extents in that dimensions
     * If a custom numDims value is specified only the last n dimensions are considered
     */
    template< typename T_Data, class T_SFINAE = void >
    struct GetExtentsRawPtr: GetExtentsRawPtrImpl< T_Data, true >{
        using Data = T_Data;
        using Parent = GetExtentsRawPtrImpl< T_Data, true >;

        using Parent::Parent;
    };

    /**
     * Specialization when we have an extents member with a data() function returning a pointer
     */
    template< typename T_Data >
    struct GetExtentsRawPtr<
        T_Data,
        std::enable_if_t<
            std::is_pointer<
                decltype(
                    std::declval<T_Data>().extents.data()
                )
            >::value
        >
    >: GetExtentsRawPtrImpl< T_Data, false >{
        using Data = T_Data;
        using Parent = GetExtentsRawPtrImpl< T_Data, false >;

        using Parent::Parent;
    };

}  // namespace policies
}  // namespace foobar
