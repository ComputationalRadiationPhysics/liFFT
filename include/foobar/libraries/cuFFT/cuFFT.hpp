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
     * Specifies that inplace transforms are only used when explicitly requested from the FFT
     */
    struct InplaceExplicit: std::false_type{};
    /**
     * Specifies that inplace transforms should be used if both memories reside on host
     */
    struct InplaceForHost: std::true_type{};

    /**
     * Wrapper for the CUDA-Library that executes the FFT on GPU(s)
     *
     * Note: Allocation and copy will only occur if the IsDeviceMemory trait returns false for the given container
     *
     * \tparam T_InplacePolicy Either InplaceExplicit or InplaceForHost
     * \tparam T_Allocator Policy to alloc/free memory for the input
     * \tparam T_Copier Policy to copy memory to and from the device (Functions H2D and D2H)
     * \tparam T_FFT_Properties Placeholder that will be replaced by a class containing the properties for this FFT
     */
    template<
        class T_InplacePolicy = InplaceForHost,
        class T_Allocator = policies::CudaAllocator,
        class T_Copier = policies::CudaMemCpy,
        class T_FFT_Properties = bmpl::_1
    >
    class CuFFT
    {
    private:
        static constexpr bool inplaceForHost = T_InplacePolicy::value;
        using Allocator = T_Allocator;
        using Copier = T_Copier;
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
        using LibTypes = traits::LibTypes< PrecisionType, FFT::isComplexIn, FFT::isComplexOut >;
        using PlanType = Plan<typename LibTypes::InType, typename LibTypes::OutType, Allocator>;

        PlanType plan_;

        CuFFT(CuFFT& obj) = delete;
        CuFFT& operator=(const CuFFT&) = delete;
    public:
        explicit CuFFT(Input& input, Output& output)
        {
            Planner()(plan_, input, output, inplaceForHost, Allocator());
        }

        explicit CuFFT(Input& inOut)
        {
            Planner()(plan_, inOut, inplaceForHost, Allocator());
        }

        CuFFT(CuFFT&& obj) = default;
        CuFFT& operator=(CuFFT&&) = default;

        void operator()(Input& input, Output& output)
        {
            Executer()(plan_, input, output, inplaceForHost, Copier());
        }

        void operator()(Input& inOut)
        {
            Executer()(plan_, inOut, inplaceForHost, Copier());
        }
    };

}  // namespace cuFFT
}  // namespace libraries
}  // namespace foobar
