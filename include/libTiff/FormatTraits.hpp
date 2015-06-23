#pragma once

#include <tiffio.h>
#include <stdint.h>
#include "libTiff/exceptions.hpp"

namespace libTiff {

    enum class ImageFormat
    {
        ARGB, // 32bit unsigned value per pixel -> 8bit per channel
        Float, // Monochrome, 32bit FP
        Double // Monochrome, 64bit FP
    };

    template< ImageFormat T_imgFormat >
    struct PixelType;

    template< ImageFormat T_imgFormat >
    struct SamplesPerPixel;

    template< ImageFormat T_imgFormat >
    struct BitsPerSample: std::integral_constant<
        uint16_t,
        sizeof(typename PixelType< T_imgFormat >::type)/SamplesPerPixel< T_imgFormat >::value
    >{};

    template<>
    struct PixelType< ImageFormat::ARGB >
    {
        using type = uint32;
        static constexpr uint16 tiffType = SAMPLEFORMAT_UINT;
    };

    template<>
    struct PixelType< ImageFormat::Float >
    {
        using type = float;
        static constexpr uint16 tiffType = SAMPLEFORMAT_IEEEFP;
    };

    template<>
    struct PixelType< ImageFormat::Double >
    {
        using type = double;
        static constexpr uint16 tiffType = SAMPLEFORMAT_IEEEFP;
    };

    template<>
    struct SamplesPerPixel< ImageFormat::ARGB >: std::integral_constant< uint16, 4 >{};

    template<>
    struct SamplesPerPixel< ImageFormat::Float >: std::integral_constant< uint16, 1 >{};

    template<>
    struct SamplesPerPixel< ImageFormat::Double >: std::integral_constant< uint16, 1 >{};

    template< ImageFormat T_imgFormat >
    bool
    needConversion(uint16 tiffSampleFormat, uint16 samplesPerPixel, uint16 bitsPerSample)
    {
        return (tiffSampleFormat != PixelType<T_imgFormat>::tiffType ||
                samplesPerPixel != SamplesPerPixel<T_imgFormat>::value ||
                bitsPerSample != BitsPerSample<T_imgFormat>::value);
    }

}  // namespace libTiff
