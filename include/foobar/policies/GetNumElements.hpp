#pragma once

#include "foobar/traits/NumDims.hpp"
#include "foobar/policies/GetExtents.hpp"
#include "foobar/policies/GetStrides.hpp"

namespace foobar {
namespace policies {

    /**
     * Returns the total number of elements for a given data structure
     *
     * \tparam T_Data structure to count
     * \tparam T_justActualElements True if just existing elements should be counted, false to account for strides
     */
    template< typename T_Data, bool T_justActualElements = true >
    struct GetNumElements
    {
        static constexpr unsigned numDims = traits::NumDims<T_Data>::value;
        using Extents = GetExtents< T_Data >;

        size_t operator()(const T_Data& data){
            Extents extents(data);
            static_assert(numDims>0, "No dimensions?");
            size_t result = 1;
            for(unsigned i=0; i<numDims; ++i)
                result *= extents[i];
            return result;
        }
    };

    template< typename T_Data >
    struct GetNumElements< T_Data, false >
    {
        static constexpr unsigned numDims = traits::NumDims<T_Data>::value;
        using Extents = GetExtents< T_Data >;
        using Strides = GetStrides< T_Data >;

        size_t operator()(const T_Data& data){
            Extents extents(data);
            Strides strides(data);
            static_assert(numDims>0, "No dimensions?");
            return strides[0] * extents[0];
        }
    };


    /**
     * Returns the total number of elements for a given data structure
     *
     * \param data structure to count
     * \param justActualElements True if just existing elements should be counted, false to account for strides (e.g. to get total mem size)
     */
    template< typename T_Data>
    size_t getNumElements(const T_Data& data, bool justActualElements = true){
        if(justActualElements)
            return GetNumElements< T_Data, true >()(data);
        else
            return GetNumElements< T_Data, false >()(data);
    }

}  // namespace policies
}  // namespace foobar
