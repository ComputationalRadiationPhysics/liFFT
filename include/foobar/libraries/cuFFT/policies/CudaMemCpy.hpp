#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "foobar/libraries/cuFFT/cudaSafeCall.hpp"

namespace foobar {
namespace libraries {
namespace cuFFT {
namespace policies {

    /**
     * Copier that uses standard cudaMemCpy
     */
    struct CudaMemCpy
    {
        template< typename T >
        void H2D(T* ptrTo, T* ptrFrom, unsigned memSize) const
        {
            CudaSafeCall(cudaMemcpy(ptrTo, ptrFrom, memSize, cudaMemcpyHostToDevice));
        }

        template< typename T >
        void D2H(T* ptrTo, T* ptrFrom, unsigned memSize) const
        {
            CudaSafeCall(cudaMemcpy(ptrTo, ptrFrom, memSize, cudaMemcpyDeviceToHost));
        }
    };

}  // namespace policies
}  // namespace cuFFT
}  // namespace libraries
}  // namespace foobar
