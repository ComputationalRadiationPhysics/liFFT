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
                strides_[0] = 1;
                for(unsigned i=0; i+1<numDims; ++i)
                    strides_[i+1] = strides_[i] * extents[i];
            }

            unsigned operator[](unsigned dimIdx) const
            {
                return strides_[dimIdx];
            }
        protected:
            types::Vec<numDims> strides_;
        };

        /**
         * Default specialization for strided types: Use strides member
         */
        template< typename T_Data >
        struct GetStrides< T_Data, true >: private boost::noncopyable
        {
            using Data = T_Data;

            GetStrides(const Data& data): data_(data){}

            unsigned operator[](unsigned dimIdx) const
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
