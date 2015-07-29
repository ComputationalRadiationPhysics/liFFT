#pragma once

#include <boost/utility.hpp>
#include "foobar/traits/IsStrided.hpp"
#include "foobar/traits/NumDims.hpp"
#include "foobar/policies/GetExtents.hpp"
#include "foobar/types/Vec.hpp"

namespace foobar {
namespace policies {

    namespace detail {

        /**
         * Default specialization for unstrided types: Use extents
         */
        template< typename T_Data, bool T_isStrided = traits::IsStrided< T_Data >::value >
        struct GetStrides: private boost::noncopyable
        {
            using Data = T_Data;
            static constexpr unsigned numDims = traits::NumDims< Data >::value;

            GetStrides(const Data& data)
            {
                GetExtents< T_Data > extents(data);
                static_assert(numDims >= 1, "No dimensions");
                strides_[numDims - 1] = 1;
                for(unsigned i=numDims - 1; i>0; --i)
                    strides_[i-1] = strides_[i] * extents[i];
            }

            size_t operator[](unsigned dimIdx) const
            {
                return strides_[dimIdx];
            }
        protected:
            types::Vec<numDims, size_t> strides_;
        };

        /**
         * Default specialization for strided types: Use strides member
         */
        template< typename T_Data >
        struct GetStrides< T_Data, true >: private boost::noncopyable
        {
            using Data = T_Data;

            GetStrides(const Data& data): data_(data){}

            size_t operator[](unsigned dimIdx) const
            {
                return data_.strides[dimIdx];
            }
        protected:
            const Data& data_;
        };

    }  // namespace detail

    /**
     * Provides a []-operator to get the strides in the specified dimension of the data object given in the constructor
     */
    template< typename T_Data >
    struct GetStrides: detail::GetStrides< T_Data >
    {
        using detail::GetStrides< T_Data >::GetStrides;
    };

}  // namespace policies
}  // namespace foobar
