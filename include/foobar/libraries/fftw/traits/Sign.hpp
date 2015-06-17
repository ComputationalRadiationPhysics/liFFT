#pragma once

#include <fftw3.h>

namespace foobar {
namespace libraries {
namespace fftw {
namespace traits{

        /**
         * Defines the sign for the given FFT type (forward/backward)
         */
        template< bool T_isFwd=true >
        struct Sign: std::integral_constant<int, FFTW_FORWARD>{};

        template<>
        struct Sign<false>: std::integral_constant<int, FFTW_BACKWARD>{};

}  // namespace traits
}  // namespace fftw
}  // namespace libraries
}  // namespace foobar
