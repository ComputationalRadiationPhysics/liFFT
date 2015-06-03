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
    struct ExecutePlan;

    template<>
    struct ExecutePlan<float>
    {
        using PlanType = typename traits::Types<float>::PlanType;

        void
        operator()(PlanType& plan)
        {
            fftwf_execute(plan);
        }
    };

    template<>
    struct ExecutePlan<double>
    {
        using PlanType = typename traits::Types<double>::PlanType;

        void
        operator()(PlanType& plan)
        {
            fftw_execute(plan);
        }
    };

} // namespace traits
}  // namespace fftw
}  // namespace libraries
}  // namespace foobar
