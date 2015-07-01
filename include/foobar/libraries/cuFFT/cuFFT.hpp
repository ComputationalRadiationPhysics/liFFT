#pragma once

#include "foobar/libraries/cuFFT/policies/CudaAllocator.hpp"
#include "foobar/libraries/cuFFT/policies/CudaMemCpy.hpp"
#include "foobar/libraries/cuFFT/policies/Planner.hpp"
#include "foobar/libraries/cuFFT/policies/ExecutePlan.hpp"
#include <boost/mpl/placeholders.hpp>

namespace bmpl = boost::mpl;

namespace foobar {
namespace libraries {
namespace cuFFT {

    /**
     * Wrapper for the CUDA-Library that executes the FFT on GPU(s)
     *
     * Note: Allocation and copy will only occur if the IsDeviceMemory trait returns false for the given container
     *
     * \tparam T_AllocatorIn Policy to alloc/free memory for the input
     * \tparam T_AllocatorOut Policy to alloc/free memory for the output (ignored for inplace transforms)
     * \tparam T_Copier Policy to copy memory to and from the device (Functions H2D and D2H)
     * \tparam T_FFT_Properties Placeholder that will be replaced by a class containing the properties for this FFT
     */
    template<
        class T_AllocatorIn = policies::CudaAllocator,
        class T_AllocatorOut = policies::CudaAllocator,
        class T_Copier = policies::CudaMemCpy,
        class T_FFT_Properties = bmpl::_1
    >
    class CuFFT: private boost::noncopyable
    {
    private:
        using FFT = T_FFT_Properties;
        using Input = typename FFT::Input;
        using Output = typename FFT::Output;
        using PrecisionType = typename FFT::PrecisionType;
        using Planner =
                policies::Planner<
                    PrecisionType,
                    foobar::types::TypePair< Input, Output >,
                    FFT::isInplace,
                    FFT::numDims,
                    FFT::isComplexIn,
                    FFT::isComplexOut,
                    FFT::isAoSIn,
                    FFT::isAoSOut,
                    FFT::isStridedIn,
                    FFT::isStridedOut
                >;
        using Executer =
                policies::ExecutePlan<
                    PrecisionType,
                    foobar::types::TypePair< Input, Output >,
                    FFT::isFwd,
                    FFT::isInplace,
                    FFT::numDims,
                    FFT::isComplexIn,
                    FFT::isComplexOut
                >;
        using PlanType = typename Planner::PlanType;

        PlanType plan_;
        T_AllocatorIn allocIn_;
        T_AllocatorOut allocOut_;
        T_Copier copy_;

    public:
        explicit CuFFT(Input& input, Output& output)
        {
            plan_ = Planner()(input, output, allocIn_, allocOut_);
        }

        explicit CuFFT(Input& inOut)
        {
            plan_ = Planner()(inOut, allocIn_, allocOut_);
        }

        CuFFT(CuFFT&& obj):
            plan_(std::move(obj.plan_)),
            allocIn_(std::move(obj.allocIn_)),
            allocOut_(std::move(obj.allocOut_)),
            copy_(std::move(obj.copy_)){}


        ~CuFFT()
        {
            cufftDestroy(plan_.plan);
            allocIn_.free(plan_.InDevicePtr);
            if(!FFT::isInplace)
                allocOut_.free(plan_.OutDevicePtr);
        }

        void operator()(Input& input, Output& output)
        {
            Executer()(plan_, input, output, copy_);
        }

        void operator()(Input& inOut)
        {
            Executer()(plan_, inOut, copy_);
        }
    };

}  // namespace cuFFT
}  // namespace libraries
}  // namespace foobar
