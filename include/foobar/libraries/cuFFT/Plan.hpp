#pragma once

namespace foobar {
namespace libraries {
namespace cuFFT {

    template< typename T_In, typename T_Out >
    struct Plan
    {
        cufftHandle plan;
        T_In* InDevicePtr;
        T_Out* OutDevicePtr;
    };

}  // namespace cuFFT
}  // namespace libraries
}  // namespace foobar
