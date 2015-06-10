#pragma once

#include "foobar/c++14_types.hpp"
#include "foobar/AutoDetect.hpp"
#include "foobar/types/InplaceType.hpp"
#include "foobar/traits/IntegralType.hpp"
#include "foobar/traits/NumDims.hpp"
#include "foobar/traits/IsComplex.hpp"
#include "foobar/traits/IsAoS.hpp"
#include "foobar/traits/IsStrided.hpp"
#include "foobar/util.hpp"

namespace foobar {
namespace detail {

    /**
     * Type that is passed to FFT-Implementations to define some properties common for all FFTs
     * Parameters are the same as for FFT itself minus the Library
     */
    template<
        typename T_Input,
        typename T_Output,
        typename T_IsFwd,
        typename T_NumDims
        >
    struct FFT_Properties
    {
    public:
        using Input = T_Input;
        static constexpr bool isInplace = traits::IsInplace<T_Output>::value;
        using Output = std::conditional_t< isInplace, Input, T_Output >;
    private:
        static constexpr bool autoDetectIsFwd = std::is_same< T_IsFwd, AutoDetect >::value;
        static constexpr bool autoDetectNumDims = std::is_same< T_NumDims, AutoDetect >::value;
        static_assert(autoDetectIsFwd || std::is_same< typename T_IsFwd::value_type, bool >::value, "Wrong argument type for IsFwd");
        static_assert(autoDetectNumDims || std::is_same< typename T_NumDims::value_type, unsigned >::value, "Wrong argument type for NumDims");

        using PrecisionTypeIn = typename traits::IntegralType<Input>::type;
        using NumDimsIn = std::conditional_t<
                            autoDetectNumDims,
                            traits::NumDims<Input>,
                            T_NumDims >;
        using PrecisionTypeOut = typename traits::IntegralType<Output>::type;
        using NumDimsOut = std::conditional_t<
                             autoDetectNumDims,
                             traits::NumDims<Output>,
                             T_NumDims >;

        static_assert( AssertValue< std::is_same< PrecisionTypeIn, PrecisionTypeOut > >::value, "Need same precision on In/Out");
        static_assert(NumDimsIn::value >= 1, "Need >= 1 dimension");
        static_assert(NumDimsOut::value >= 1, "Need >= 1 dimension");
        static_assert(NumDimsIn::value == NumDimsOut::value, "Dimension mismatch");
    public:
        static constexpr bool isComplexIn = traits::IsComplex<Input>::value;
        static constexpr bool isAoSIn = traits::IsAoS<Input>::value;
        static constexpr bool isStridedIn = traits::IsStrided<Input>::value;

        static constexpr bool isComplexOut = traits::IsComplex< std::conditional_t< isInplace, T_Output, Output > >::value;
        static constexpr bool isAoSOut = traits::IsAoS<Output>::value;
        static constexpr bool isStridedOut = traits::IsStrided<Output>::value;

        using PrecisionType = PrecisionTypeIn;
        static constexpr bool isAoS = isAoSIn || isAoSOut;
        static constexpr bool isStrided = isStridedIn || isStridedOut;

        static constexpr unsigned numDims = NumDimsIn::value;
        static constexpr bool isFwd = (autoDetectIsFwd && isComplexOut) || (!autoDetectIsFwd && T_IsFwd::value);
    };

}  // namespace detail
}  // namespace foobar
