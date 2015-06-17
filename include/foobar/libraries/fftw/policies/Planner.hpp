#pragma once

#include <cassert>
#include "foobar/types/TypePair.hpp"
#include "foobar/policies/GetExtentsRawPtr.hpp"
#include "foobar/policies/GetRawPtr.hpp"
#include "foobar/policies/Ptr2Ptr.hpp"
#include "foobar/libraries/fftw/traits/Sign.hpp"
#include "foobar/libraries/fftw/policies/CreatePlan.hpp"
#include "foobar/c++14_types.hpp"

namespace foobar {
namespace libraries {
namespace fftw {
namespace policies {

    template<
            typename T_Precision,
            class T_InOutTypes,
            bool T_isFwd,
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
     * Both AoS, Both non-strided
     */
    template< typename T_Precision, class T_InOutTypes, bool T_isFwd, bool T_isInplace, unsigned T_numDims, bool T_isComplexIn, bool T_isComplexOut >
    struct Planner< T_Precision, T_InOutTypes, T_isFwd, T_isInplace, T_numDims, T_isComplexIn, T_isComplexOut, true, true, false, false >
    {
    private:
        using Precision = T_Precision;
        static constexpr unsigned numDims = T_numDims;
        using Input  = typename T_InOutTypes::First;
        using Output = typename T_InOutTypes::Second;
        static constexpr bool isFwd        = T_isFwd;
        static constexpr bool isInplace    = T_isInplace;
        static constexpr bool isComplexIn  = T_isComplexIn;
        static constexpr bool isComplexOut = T_isComplexOut;
        using ExtentsIn = foobar::policies::GetExtentsRawPtr< Input >;
        using ExtentsOut = foobar::policies::GetExtentsRawPtr< Output >;
        using RawPtrIn = foobar::policies::GetRawPtr< Input >;
        using RawPtrOut = foobar::policies::GetRawPtr< Output >;
        using ComplexType = typename traits::Types< Precision >::ComplexType;
        using PtrConverterIn = std::conditional_t<
                                    isComplexIn,
                                    foobar::policies::Ptr2Ptr<Precision, ComplexType>,
                                    foobar::policies::Ptr2Ptr<Precision>
                                >;
        using PtrConverterOut = std::conditional_t<
                                    isComplexOut,
                                    foobar::policies::Ptr2Ptr<Precision, ComplexType>,
                                    foobar::policies::Ptr2Ptr<Precision>
                                >;
        static_assert(isComplexIn || isComplexOut, "Real2Real transform not supported");
        static_assert(isComplexIn || isFwd, "Real2Complex is always a forward transform");
        static_assert(isComplexOut || !isFwd, "Complex2Real is always a backward transform");

    public:
        using PlanType = typename traits::Types<T_Precision>::PlanType;

        PlanType
        operator()(Input& input, Output& output, const unsigned flags = FFTW_ESTIMATE)
        {
            static_assert(!isInplace, "Cannot be used for inplace transforms!");
            ExtentsIn extents(input);
            ExtentsOut extentsOut(output);
            for(unsigned i=0; i<numDims; ++i){
                unsigned eIn = extents()[i];
                unsigned eOut = extentsOut()[i];
                // Same extents in all dimensions unless we have a C2R or R2C and compare the last dimension
                assert(eIn == eOut || (i+1 == numDims && !(isComplexIn && isComplexOut)));
                // Half input size for first dimension of R2C
                assert(isComplexIn || i+1 != numDims || eIn/2+1 == eOut);
                // Half output size for first dimension of C2R
                assert(isComplexOut || i+1 != numDims || eIn == eOut/2+1);
            }
            return policies::CreatePlan<Precision>().Create(
                    numDims,
                    reinterpret_cast<const int*>(extents()),
                    PtrConverterIn()(RawPtrIn()(input)),
                    PtrConverterOut()(RawPtrOut()(output)),
                    traits::Sign<isFwd>::value,
                    flags
                    );
        }

        PlanType
        operator()(Input& inOut, const unsigned flags = FFTW_ESTIMATE)
        {
            static_assert(isInplace, "Must be used for inplace transforms!");
            ExtentsIn extents(inOut);
            RawPtrIn inPtr;
            return policies::CreatePlan<Precision>().Create(
                    numDims,
                    reinterpret_cast<const int*>(extents()),
                    PtrConverterIn()(inPtr(inOut)),
                    PtrConverterOut()(inPtr(inOut)),
                    traits::Sign<isFwd>::value,
                    flags
                    );
        }
    };

}  // namespace policies
}  // namespace fftw
}  // namespace libraries
}  // namespace foobar
