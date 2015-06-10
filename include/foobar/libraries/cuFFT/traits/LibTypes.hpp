#pragma once

#include "foobar/libraries/cuFFT/traits/Types.hpp"
#include "foobar/c++14_types.hpp"

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
        using InType = std::conditional_t<
                              isComplexIn,
                              ComplexType,
                              RealType
                          >;
        using OutType = std::conditional_t<
                              isComplexOut,
                              ComplexType,
                              RealType
                          >;
    };

}  // namespace traits
}  // namespace cuFFT
}  // namespace libraries
}  // namespace foobar
