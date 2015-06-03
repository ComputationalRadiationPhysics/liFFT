#pragma once

#include "foobar/traits/IntegralType.hpp"
#include "foobar/traits/IsComplex.hpp"
#include "foobar/traits/IsAoS.hpp"
#include "foobar/traits/IsStrided.hpp"
#include "foobar/traits/MemoryType.hpp"
#include "foobar/traits/NumDims.hpp"
#include "foobar/types/InplaceType.hpp"

namespace foobar {
namespace policies {

    struct AutoDetect: std::integral_constant<unsigned, 0>{};

    template<
            template< typename > class T_Library,
            bool T_isFwd,
            typename T_Input,
            typename T_Output = types::InplaceType<>,
            typename T_NumDims = AutoDetect
            >
    struct FFT
    {
    public:
        static constexpr bool isFwd = T_isFwd;
        using Input = T_Input;
        static constexpr bool isInplace = traits::IsInplace<T_Output>::value;
        using Output = typename std::conditional< isInplace, Input, T_Output >::type;

        using MemIn = typename traits::MemoryType<Input>::type;
        using MemOut = typename traits::MemoryType<Output>::type;
    private:
        static constexpr bool autoDetectNumDims = std::is_same< T_NumDims, AutoDetect >::value;
        static_assert(!autoDetectNumDims,"???");

        using PrecisionTypeIn = typename traits::IntegralType<MemIn>::type;
        using NumDimsIn = typename std::conditional<
                            autoDetectNumDims,
                            traits::NumDims<Input>,
                            T_NumDims >::type;
        using PrecisionTypeOut = typename traits::IntegralType<MemOut>::type;
        using NumDimsOut = typename std::conditional<
                             autoDetectNumDims,
                             traits::NumDims<Output>,
                             T_NumDims >::type;

        static_assert(std::is_same< PrecisionTypeIn, PrecisionTypeOut >::value, "Need same precision on In/Out");
        static_assert(NumDimsIn::value >= 1, "Need >= 1 dimension");
        static_assert(NumDimsOut::value >= 1, "Need >= 1 dimension");
        static_assert(NumDimsIn::value == NumDimsOut::value, "Dimension mismatch");
    public:
        static constexpr bool isComplexIn = traits::IsComplex<MemIn>::value;
        static constexpr bool isAoSIn = traits::IsAoS<MemIn>::value;
        static constexpr bool isStridedIn = traits::IsStrided<MemIn>::value;

        static constexpr bool isComplexOut = traits::IsComplex< typename std::conditional< isInplace, T_Output, MemOut >::type >::value;
        static constexpr bool isAoSOut = traits::IsAoS<MemOut>::value;
        static constexpr bool isStridedOut = traits::IsStrided<MemOut>::value;

        using PrecisionType = PrecisionTypeIn;
        static constexpr bool isAoS = isAoSIn || isAoSOut;
        static constexpr bool isStrided = isStridedIn || isStridedOut;

        static constexpr unsigned numDims = NumDimsIn::value;

        using type = T_Library< FFT< T_Library, T_isFwd, T_Input, T_Output, T_NumDims > >;
    };

}  // namespace policies
}  // namespace foobar
