#pragma once

#include <cassert>
#include "foobar/types/TypePair.hpp"
#include "foobar/policies/GetExtentsRawPtr.hpp"
#include "foobar/policies/GetRawPtr.hpp"
#include "foobar/policies/Ptr2Ptr.hpp"
#include "foobar/libraries/fftw/traits/Sign.hpp"
#include "foobar/libraries/fftw/policies/CreatePlan.hpp"

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
    struct Planner< T_Precision, T_InOutTypes, T_isFwd, T_isInplace, T_numDims, T_isComplexIn, T_isComplexOut, T_isComplexIn, T_isComplexOut, false, false >
    {
        using PlanType = typename traits::Types<T_Precision>::PlanType;
        static constexpr unsigned numDims = T_numDims;
        using Input = typename T_InOutTypes::First;
        using Output = typename T_InOutTypes::Second;
        using ExtendsIn = foobar::policies::GetExtentsRawPtr< Input, numDims >;
        using ExtendsOut = foobar::policies::GetExtentsRawPtr< Output, numDims >;
        using RawPtrIn = foobar::policies::GetRawPtr< Input >;
        using RawPtrOut = foobar::policies::GetRawPtr< Output >;
        using ComplexType = typename traits::Types< T_Precision >::ComplexType;
        using PtrConverterIn = typename std::conditional<
                                    T_isComplexIn,
                                    foobar::policies::Ptr2Ptr<T_Precision, ComplexType>,
                                    foobar::policies::Ptr2Ptr<T_Precision>
                                >::type;
        using PtrConverterOut = typename std::conditional<
                                    T_isComplexOut,
                                    foobar::policies::Ptr2Ptr<T_Precision, ComplexType>,
                                    foobar::policies::Ptr2Ptr<T_Precision>
                                >::type;
        static_assert(T_isComplexIn || T_isComplexOut, "Real2Real conversion not supported");

        PlanType
        operator()(const Input& input, Output& output, const unsigned flags = FFTW_ESTIMATE)
        {
            static_assert(!T_isInplace, "Cannot be used for inplace transforms!");
            ExtendsIn extends(input);
            ExtendsOut extendsOut(output);
            for(unsigned i=0; i<T_numDims; ++i){
                unsigned eIn = extends()[i];
                unsigned eOut = extendsOut()[i];
                // Same extends in all dimensions unless we have a C2R or R2C and compare the first dimension
                assert(eIn == eOut || (i == 0 && !(T_isComplexIn && T_isComplexOut)));
                // Half input size for first dimension of R2C
                assert(eIn/2+1 == eOut || i != 0 || T_isComplexIn);
                // Half output size for first dimension of C2R
                assert(eIn == eOut/2+1 || i != 0 || T_isComplexOut);
            }
            RawPtrIn inPtr(input);
            RawPtrOut outPtr(output);
            return policies::CreatePlan<T_Precision>().Create(
                    T_numDims,
                    reinterpret_cast<const int*>(extends()),
                    PtrConverterIn()(inPtr()),
                    PtrConverterOut()(outPtr()),
                    traits::Sign<T_isFwd>::value,
                    flags
                    );
        }

        PlanType
        operator()(const Input& inOut, const unsigned flags = FFTW_ESTIMATE)
        {
            static_assert(T_isInplace, "Must be used for inplace transforms!");
            ExtendsIn extends(inOut);
            RawPtrIn inPtr(inOut);
            return policies::CreatePlan<T_Precision>().Create(
                    T_numDims,
                    reinterpret_cast<const int*>(extends()),
                    PtrConverterIn()(inPtr()),
                    PtrConverterOut()(inPtr()),
                    traits::Sign<T_isFwd>::value,
                    flags
                    );
        }
    };

}  // namespace policies
}  // namespace fftw
}  // namespace libraries
}  // namespace foobar
