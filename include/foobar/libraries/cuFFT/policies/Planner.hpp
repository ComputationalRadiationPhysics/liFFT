#pragma once

#include <cassert>
#include <cufft.h>
#include "foobar/types/TypePair.hpp"
#include "foobar/policies/GetExtents.hpp"
#include "foobar/policies/GetNumElements.hpp"
#include "foobar/policies/GetRawPtr.hpp"
#include "foobar/libraries/cuFFT/Plan.hpp"
#include "foobar/libraries/cuFFT/traits/FFTType.hpp"
#include "foobar/libraries/cuFFT/traits/Sign.hpp"
#include "foobar/libraries/cuFFT/traits/LibTypes.hpp"

namespace foobar {
namespace libraries {
namespace cuFFT {

namespace policies {

    /**
     * Functor that creates a plan.
     * Takes references to the plan, input, output and their allocators
     */
    template<
            typename T_Precision,
            class T_InOutTypes,
            bool T_isInplace,
            unsigned T_numDims,
            bool T_isComplexIn,
            bool T_isComplexOut,
            bool T_isAoSIn,
            bool T_isAoSOut,
            bool T_isStridedIn,
            bool T_isStridedOut
    >
    struct Planner;

    /**
     * Both AoS (if complex), Both non-strided
     */
    template<
            typename T_Precision,
            class T_InOutTypes,
            bool T_isInplace,
            unsigned T_numDims,
            bool T_isComplexIn,
            bool T_isComplexOut
    >
    struct Planner< T_Precision, T_InOutTypes, T_isInplace, T_numDims, T_isComplexIn, T_isComplexOut, T_isComplexIn, T_isComplexOut, false, false >
    {
    private:
        using Precision = T_Precision;
        static constexpr unsigned numDims = T_numDims;
        using Input  = typename T_InOutTypes::First;
        using Output = typename T_InOutTypes::Second;
        static constexpr bool isInplace    = T_isInplace;
        static constexpr bool isComplexIn  = T_isComplexIn;
        static constexpr bool isComplexOut = T_isComplexOut;

        using ExtentsIn = foobar::policies::GetExtents< Input >;
        using ExtentsOut = foobar::policies::GetExtents< Output >;
        using LibTypes = traits::LibTypes< Precision, isComplexIn, isComplexOut >;
        using LibInType = typename LibTypes::InType;
        using LibOutType = typename LibTypes::OutType;
        using FFTType = traits::FFTType< Precision, isComplexIn, isComplexOut >;

        static_assert(isComplexIn || isComplexOut, "Real2Real conversion not supported");
        //static_assert(T_isComplexIn || T_isFwd, "Real2Complex is always a forward transform");
        //static_assert(T_isComplexOut || !T_isFwd, "Complex2Real is always a backward transform");
        static_assert(numDims > 0 && numDims <= 3, "Only 1D-3D is supported");

    public:
        using PlanType = Plan< LibInType, LibOutType >;

    private:
        template< class T_Extents >
        void
        createPlan(PlanType& plan, T_Extents& extents)
        {
            cufftResult result;
            if(numDims == 1)
                result = cufftPlan1d(&plan.plan, extents[0], FFTType::value, 1);
            else if(numDims == 2)
                result = cufftPlan2d(&plan.plan, extents[0], extents[1], FFTType::value);
            else
                result = cufftPlan3d(&plan.plan, extents[0], extents[1], extents[2], FFTType::value);
            if(result != CUFFT_SUCCESS)
                throw std::runtime_error("Error creating plan: " + std::to_string(result));
        }
    public:

        template< class T_AllocatorIn, class T_AllocatorOut >
        PlanType
        operator()(Input& input, Output& output, T_AllocatorIn allocIn, T_AllocatorOut allocOut)
        {
            static_assert(!isInplace, "Cannot be used for inplace transforms!");
            ExtentsIn extents(input);
            ExtentsOut extentsOut(output);
            for(unsigned i=0; i<numDims; ++i){
                unsigned eIn = extents[i];
                unsigned eOut = extentsOut[i];
                // Same extents in all dimensions unless we have a C2R or R2C and compare the last dimension
                assert(eIn == eOut || (i+1 == numDims && !(isComplexIn && isComplexOut)));
                // Half input size for first dimension of R2C
                assert(isComplexIn || i+1 != numDims || eIn/2+1 == eOut);
                // Half output size for first dimension of C2R
                assert(isComplexOut || i+1 != numDims || eIn == eOut/2+1);
            }
            PlanType plan;
            createPlan(plan, extents);
            unsigned numElements = foobar::policies::GetNumElements< Input >()(input);
            allocIn.malloc(plan.InDevicePtr, numElements * sizeof(LibInType));
            allocOut.malloc(plan.OutDevicePtr, numElements * sizeof(LibOutType));
            return plan;
        }

        template< class T_Allocator >
        PlanType
        operator()(Input& inOut, T_Allocator alloc)
        {
            static_assert(isInplace, "Must be used for inplace transforms!");
            ExtentsIn extents(inOut);
            PlanType plan;
            createPlan(plan, extents);
            unsigned numElements = foobar::policies::GetNumElements< ExtentsIn >()(extents);
            alloc.malloc(plan.InDevicePtr, numElements * std::max(sizeof(LibInType), sizeof(LibOutType)));
            plan.OutDevicePtr = nullptr;
            return plan;
        }
    };

}  // namespace policies
}  // namespace cuFFT
}  // namespace libraries
}  // namespace foobar
