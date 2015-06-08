#pragma once

#include "foobar/libraries/cuFFT/traits/Types.hpp"

namespace foobar {
namespace libraries {
namespace cuFFT {
namespace traits{

    /**
     * Defines the LibIn/OutType for a configuration
     */
    template<
            typename T_Precision,
            bool T_isComplexIn,
            bool T_isComplexOut
    >
    struct LibTypes
    {
        using Precision = T_Precision;
        static constexpr bool isComplexIn = T_isComplexIn;
        static constexpr bool isComplexOut = T_isComplexOut;

        using RealType = typename Types< Precision >::RealType;
        using ComplexType = typename Types< Precision >::ComplexType;
        using InType = typename std::conditional<
                              isComplexIn,
                              ComplexType,
                              RealType
                          >::type;
        using OutType = typename std::conditional<
                              isComplexOut,
                              ComplexType,
                              RealType
                          >::type;
    };

}  // namespace traits
}  // namespace cuFFT
}  // namespace libraries
}  // namespace foobar
