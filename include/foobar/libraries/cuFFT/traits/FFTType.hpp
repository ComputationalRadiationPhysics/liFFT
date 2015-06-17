#pragma once

#include <cufft.h>

namespace foobar {
namespace libraries {
namespace cuFFT {
namespace traits{

        /**
         * Defines the (cuFFT)-Type of the FFT
         */
        template< typename T_Precision, bool T_IsComplexIn, bool T_IsComplexOut >
        struct FFTType: std::integral_constant< cufftType, CUFFT_C2C >{};

        template<>
        struct FFTType< float, false, true >: std::integral_constant< cufftType, CUFFT_R2C >{};

        template<>
        struct FFTType< float, true, false >: std::integral_constant< cufftType, CUFFT_C2R >{};

        template<>
        struct FFTType< double, true, true >: std::integral_constant< cufftType, CUFFT_Z2Z >{};

        template<>
        struct FFTType< double, false, true >: std::integral_constant< cufftType, CUFFT_D2Z >{};

        template<>
        struct FFTType< double, true, false >: std::integral_constant< cufftType, CUFFT_Z2D >{};

}  // namespace traits
}  // namespace cuFFT
}  // namespace libraries
}  // namespace foobar
