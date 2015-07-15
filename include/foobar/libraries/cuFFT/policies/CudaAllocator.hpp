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
        T*
        malloc(size_t memSize) const
        {
            void* ptr;
            CudaSafeCall(cudaMalloc(&ptr, memSize));
            return reinterpret_cast<T*>(ptr);
        }

        template< typename T >
        void
        free(T* ptr) const
        {
            CudaSafeCall(cudaFree(ptr));
        }
    };

}  // namespace policies
}  // namespace cuFFT
}  // namespace libraries
}  // namespace foobar
