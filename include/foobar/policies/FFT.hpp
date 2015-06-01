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

    template<
            template< typename > class T_Library,
            bool T_isFwd,
            typename T_Input,
            typename T_Output = types::InplaceType<>
            >
    struct FFT: public T_Library< FFT >::type
    {
        static constexpr bool isFwd = T_isFwd;
        using Input = T_Input;
        static constexpr bool isInplace = traits::IsInplace<T_Output>;
        using Output = typename std::conditional< isInplace, Input, T_Output >::type;

        using MemIn = typename traits::MemoryType<Input>::type;
        using PrecisionTypeIn = typename traits::IntegralType<MemIn>::type;
        static constexpr bool isComplexIn = traits::IsComplex<MemIn>::value;
        static constexpr unsigned numDimsIn = traits::NumDims<Input>::value;
        static constexpr bool isAoSIn = traits::IsAoS<MemIn>::value;
        static constexpr bool isStridedIn = traits::IsStrided<MemIn>::value;

        using MemOut = typename traits::MemoryType<Output>::type;
        using PrecisionTypeOut = typename traits::IntegralType<MemOut>::type;
        static constexpr bool isComplexOut = traits::IsComplex< typename std::conditional< isInplace, T_Output, MemOut >::type >::value;
        static constexpr unsigned numDimsOut = traits::NumDims<Output>::value;
        static constexpr bool isAoSOut = traits::IsAoS<MemOut>::value;
        static constexpr bool isStridedOut = traits::IsStrided<MemOut>::value;

        using PrecisionType = PrecisionTypeIn;
        static constexpr bool isAoS = isAoSIn || isAoSOut;
        static constexpr bool isStrided = isStridedIn || isStridedOut;

        static constexpr unsigned numDims = numDimsIn;

        static_assert(std::is_same< PrecisionTypeIn, PrecisionTypeOut >::value, "Need same precision on In/Out");
        static_assert(numDimsIn >= 1, "Need >= 1 dimension");
        static_assert(numDimsOut >= 1, "Need >= 1 dimension");
        static_assert(numDimsIn == numDimsOut, "Dimension mismatch");
    };

}  // namespace policies
}  // namespace foobar
