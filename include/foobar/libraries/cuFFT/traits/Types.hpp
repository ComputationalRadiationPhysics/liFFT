#pragma once

namespace foobar {
namespace libraries {
namespace cuFFT {
namespace traits{

    /**
     * Defines the FFTW internal types for the given precision type:
     * PlanType, ComplexType
     */
    template< typename T_Precision >
    struct Types;

    template<>
    struct Types<float>
    {
        using ComplexType = cufftComplex;
        using RealType = cufftReal;
    };

    template<>
    struct Types<double>
    {
        using ComplexType = cufftDoubleComplex;
        using RealType = cufftDoubleReal;
    };

}  // namespace traits
}  // namespace cuFFT
}  // namespace libraries
}  // namespace foobar
