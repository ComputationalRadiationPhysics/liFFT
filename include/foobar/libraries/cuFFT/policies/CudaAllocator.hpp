#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "foobar/libraries/cuFFT/cudaSafeCall.hpp"

namespace foobar {
namespace libraries {
namespace cuFFT {
namespace policies {

    /**
     * Allocator that uses cudaMalloc/cudaFree
     */
    struct CudaAllocator
    {
        template< typename T >
        void malloc(T*& ptr, unsigned memSize)
        {
            CudaSafeCall(cudaMalloc(reinterpret_cast<void**>(&ptr), memSize));
        }

        template< typename T >
        void free(T* ptr)
        {
            CudaSafeCall(cudaFree(ptr));
        }
    };

}  // namespace policies
}  // namespace cuFFT
}  // namespace libraries
}  // namespace foobar
