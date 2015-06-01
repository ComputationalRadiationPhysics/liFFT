#pragma once

#include <fftw3.h>
#include "foobar/libraries/fftw/traits/Types.hpp"

namespace foobar {
namespace libraries {
namespace fftw {
namespace policies{

    /**
     * Transforms a pointer to the given precision type (float, double)
     * to the correct complex type required by FFTW
     */
    template< typename T_Precision >
    struct Ptr2ComplexPtr;

    namespace detail {


        template< typename T_Precision >
        struct Ptr2ComplexPtr
        {
            using ComplexType = typename traits::Types< T_Precision >::ComplexType;

            ComplexType*
            operator()(T_Precision* data)
            {
                return reinterpret_cast<ComplexType*>(data);
            }

            const ComplexType*
            operator()(const T_Precision* data)
            {
                return reinterpret_cast<ComplexType*>(data);
            }
        };

    }  // namespace detail

    template
    struct Ptr2ComplexPtr<float>: detail::Ptr2ComplexPtr<float>{};

    template
    struct Ptr2ComplexPtr<double>: detail::Ptr2ComplexPtr<double>{};


    template< typename T_Precision >
    auto FFTW_ptr2ComplexPtr(T_Precision* data)
    {
        return Ptr2ComplexPtr<T_Precision>()(data);
    }

    template< typename T_Precision >
    auto FFTW_ptr2ComplexPtr(const T_Precision* data)
    {
        return Ptr2ComplexPtr<T_Precision>()(data);
    }

} // namespace traits
}  // namespace fftw
}  // namespace libraries
}  // namespace foobar
