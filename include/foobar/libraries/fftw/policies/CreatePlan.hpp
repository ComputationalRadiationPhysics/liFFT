#pragma once

#include <fftw3.h>
#include "foobar/libraries/fftw/traits/Types.hpp"

namespace foobar {
namespace libraries {
namespace fftw {
namespace policies{

    /**
     * Creates a plan for the given precision type
     */
    template< typename T_Precision >
    struct CreatePlan;

    template<>
    struct CreatePlan<float>
    {
        using PlanType = typename traits::Types<float>::PlanType;
        using ComplexType = typename traits::Types<float>::ComplexType;
        using RealType = float;

        static_assert(!std::is_same<RealType, ComplexType>::value, "Need different types for Real/Complex");

        PlanType
        Create(int rank, const int* n, ComplexType* in, ComplexType* out, int sign, unsigned flags)
        {
            return fftwf_plan_dft(rank, n, in, out, sign, flags);
        }

        PlanType
        Create(int rank, const int* n, RealType* in, ComplexType* out, int sign, unsigned flags)
        {
            assert(sign == FFTW_FORWARD);
            return fftwf_plan_dft_r2c(rank, n, in, out, flags);
        }

        PlanType
        Create(int rank, const int* n, ComplexType* in, RealType* out, int sign, unsigned flags)
        {
            assert(sign == FFTW_BACKWARD);
            return fftwf_plan_dft_c2r(rank, n, in, out, flags);
        }
    };

    template<>
    struct CreatePlan<double>
    {
        using PlanType = typename traits::Types<double>::PlanType;
        using ComplexType = typename traits::Types<double>::ComplexType;
        using RealType = double;

        static_assert(!std::is_same<RealType, ComplexType>::value, "Need different types for Real/Complex");

        PlanType
        Create(int rank, const int* n, ComplexType* in, ComplexType* out, int sign, unsigned flags)
        {
            return fftw_plan_dft(rank, n, in, out, sign, flags);
        }

        PlanType
        Create(int rank, const int* n, RealType* in, ComplexType* out, int sign, unsigned flags)
        {
            assert(sign == FFTW_FORWARD);
            return fftw_plan_dft_r2c(rank, n, in, out, flags);
        }

        PlanType
        Create(int rank, const int* n, ComplexType* in, RealType* out, int sign, unsigned flags)
        {
            assert(sign == FFTW_BACKWARD);
            return fftw_plan_dft_c2r(rank, n, in, out, flags);
        }
    };

}  // namespace traits
}  // namespace fftw
}  // namespace libraries
}  // namespace foobar
