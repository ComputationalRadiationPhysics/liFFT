#pragma once

#include "foobar/FFT_Kind.hpp"
#include "foobar/AutoDetect.hpp"
#include "foobar/c++14_types.hpp"

namespace foobar {

    /**
     * Defines a FFT
     *
     * \tparam T_Library FFT-Library to use
     * \tparam T_kind Type of FFT to execute (Complex-Complex, Real-Complex...)
     * \tparam T_numDims Dimensionality of the FFT
     * \tparam T_PrecisionType Base type to use (float, double, ...)
     * \tparam T_IsFwd Whether to perform a forward FFT. Can be left at AutoDetect so it is true for Real-Complex and false for Complex-Real
     * \tparam T_isInplace Whether to perform the FFT inplace, that is, it does not allocate separate memory for the output
     */
    template<
        FFT_Kind T_kind,
        unsigned T_numDims,
        typename T_PrecisionType,
        class T_IsFwd = AutoDetect,
        bool T_isInplace = false
        >
    struct FFT_Definition
    {
        static constexpr FFT_Kind kind = T_kind;
        static constexpr unsigned numDims = T_numDims;
        using PrecisionType = T_PrecisionType;
        using IsFwd = T_IsFwd;
        static constexpr bool isInplace = T_isInplace;

        static constexpr bool autoDetectIsFwd = std::is_same< T_IsFwd, AutoDetect >::value;
        static_assert(
                !autoDetectIsFwd || kind!=FFT_Kind::Complex2Complex,
                "Auto-detect is not possible for Complex-Complex transforms");

        static constexpr bool isComplexInput = traits::IsComplexInput<kind>::value;
        static constexpr bool isComplexOutput = traits::IsComplexOutput<kind>::value;
        static constexpr bool isFwd = (autoDetectIsFwd && isComplexOutput) || (!autoDetectIsFwd && IsFwd::value);
    };

    // Some definitions for commonly used FFT types

    template< typename T_Precision = float >
    using FFT_2D_C2C = FFT_Definition< FFT_Kind::Complex2Complex, 2, T_Precision, std::true_type >;
    using FFT_2D_C2C_F = FFT_2D_C2C< float >;
    using FFT_2D_C2C_D = FFT_2D_C2C< double >;

    template< typename T_Precision = float >
    using FFT_2D_R2C = FFT_Definition< FFT_Kind::Real2Complex, 2, T_Precision >;
    using FFT_2D_R2C_F = FFT_2D_R2C< float >;
    using FFT_2D_R2C_D = FFT_2D_R2C< double >;

    template< typename T_Precision = float >
    using FFT_2D_C2R = FFT_Definition< FFT_Kind::Complex2Real, 2, T_Precision >;
    using FFT_2D_C2R_F = FFT_2D_C2R< float >;
    using FFT_2D_C2R_D = FFT_2D_C2R< double >;

    template< typename T_Precision = float >
    using FFT_3D_C2C = FFT_Definition< FFT_Kind::Complex2Complex, 3, T_Precision, std::true_type >;
    using FFT_3D_C2C_F = FFT_3D_C2C< float >;
    using FFT_3D_C2C_D = FFT_3D_C2C< double >;

    template< typename T_Precision = float >
    using FFT_3D_R2C = FFT_Definition< FFT_Kind::Real2Complex, 3, T_Precision >;
    using FFT_3D_R2C_F = FFT_3D_R2C< float >;
    using FFT_3D_R2C_D = FFT_3D_R2C< double >;

    template< typename T_Precision = float >
    using FFT_3D_C2R = FFT_Definition< FFT_Kind::Complex2Real, 3, T_Precision >;
    using FFT_3D_C2R_F = FFT_3D_C2R< float >;
    using FFT_3D_C2R_D = FFT_3D_C2R< double >;

}  // namespace foobar
