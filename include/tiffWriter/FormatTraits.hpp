/* This file is part of libLiFFT.
 *
 * libLiFFT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * libLiFFT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with libLiFFT.  If not, see <www.gnu.org/licenses/>.
 */
 
#pragma once

#include <tiffio.h>
#include <stdint.h>
#include "tiffWriter/exceptions.hpp"
#include "tiffWriter/ImageFormat.hpp"

namespace tiffWriter {

    template< ImageFormat T_imgFormat >
    struct PixelType;

    template< ImageFormat T_imgFormat >
    struct SamplesPerPixel;

    template< ImageFormat T_imgFormat >
    struct BitsPerSample: std::integral_constant<
        uint16_t,
        sizeof(typename PixelType< T_imgFormat >::type)/SamplesPerPixel< T_imgFormat >::value * 8
    >{};

    template<>
    struct PixelType< ImageFormat::ARGB >
    {
        using type = uint32;
        using ChannelType = uint8_t;
        static constexpr uint16 tiffType = SAMPLEFORMAT_UINT;
    };

    template<>
    struct PixelType< ImageFormat::Float >
    {
        using type = float;
        using ChannelType = float;
        static constexpr uint16 tiffType = SAMPLEFORMAT_IEEEFP;
    };

    template<>
    struct PixelType< ImageFormat::Double >
    {
        using type = double;
        using ChannelType = double;
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

}  // namespace tiffWriter
