#pragma once

#include <fftw3.h>
#include "foobar/libraries/fftw/traits/Types.hpp"

namespace foobar {
namespace libraries {
namespace fftw {
namespace policies{

    /**
     * Executes a given plan
     */
    template< typename T_Precision >
    struct CreatePlan;

    template
    struct CreatePlan<float>
    {
        using PlanType = typename traits::Types<float>::PlanType;
        using ComplexType = typename traits::Types<float>::ComplexType;

        PlanType
        CreateC2C(int rank, const int* n, ComplexType* in, ComplexType* out, int sign, unsigned flags)
        {
            return fftwf_plan_dft(rank, n, in, out, sign, flags);
        }
    };

    template
    struct CreatePlan<double>
    {
        using PlanType = typename traits::Types<double>::PlanType;
        using ComplexType = typename traits::Types<double>::ComplexType;

        PlanType
        CreateC2C(int rank, const int* n, ComplexType* in, ComplexType* out, int sign, unsigned flags)
        {
            return fftw_plan_dft(rank, n, in, out, sign, flags);
        }
    };

} // namespace traits
}  // namespace fftw
}  // namespace libraries
}  // namespace foobar
