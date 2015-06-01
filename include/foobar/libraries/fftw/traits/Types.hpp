#pragma once

namespace foobar {
namespace libraries {
namespace fftw {
namespace traits{

    /**
     * Defines the FFTW internal types for the given precision type:
     * PlanType, ComplexType
     */
    template< typename T_Precision >
    struct Types;

    template
    struct Types<float>
    {
        using PlanType = fftwf_plan;
        using ComplexType = fftwf_complex;
    };

    template
    struct Types<double>
    {
        using PlanType = fftw_plan;
        using ComplexType = fftw_complex;
    };

} // namespace traits
}  // namespace fftw
}  // namespace libraries
}  // namespace foobar
