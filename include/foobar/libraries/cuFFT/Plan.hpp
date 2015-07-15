#pragma once

#include <memory>

namespace foobar {
namespace libraries {
namespace cuFFT {

    template< typename T_In, typename T_Out, class T_Deleter >
    struct Plan
    {
    private:
        template<typename T>
        struct Deleter
        {
            void
            operator()(T* ptr)
            {
                T_Deleter().free(ptr);
            }
        };
    public:
        cufftHandle plan;
        std::unique_ptr<T_In, Deleter<T_In>> InDevicePtr;
        std::unique_ptr<T_Out, Deleter<T_Out>> OutDevicePtr;
  };

}  // namespace cuFFT
}  // namespace libraries
}  // namespace foobar
