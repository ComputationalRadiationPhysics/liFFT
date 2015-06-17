#pragma once

#include <cufft.h>

namespace foobar {
namespace libraries {
namespace cuFFT {
namespace traits{

        /**
         * Defines the sign for the given FFT type (forward/backward)
         */
        template< bool T_isFwd=true >
        struct Sign: std::integral_constant<int, CUFFT_FORWARD>{};

        template<>
        struct Sign<false>: std::integral_constant<int, CUFFT_INVERSE>{};

}  // namespace traits
}  // namespace cuFFT
}  // namespace libraries
}  // namespace foobar
