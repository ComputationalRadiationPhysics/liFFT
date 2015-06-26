#pragma once

namespace libTiff{

    enum class ImageFormat
    {
        ARGB, // 32bit unsigned value per pixel -> 8bit per channel
        Float, // Monochrome, 32bit FP
        Double // Monochrome, 64bit FP
    };

}  // namespace name
