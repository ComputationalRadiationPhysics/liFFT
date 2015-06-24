#pragma once

#include <string>
#include <cassert>
#include <tiffio.h>
#include <boost/utility.hpp>
#include "libTiff/FormatTraits.hpp"

namespace libTiff
{
    /**
     * Allocator that uses the TIFF functions
     */
    struct TiffAllocator
    {
        template< typename T >
        void
        malloc(T*& p, size_t size)
        {
            p = static_cast<T*>(_TIFFmalloc(size));
        }

        template< typename T >
        void
        free(T* ptr)
        {
            _TIFFfree(ptr);
        }
    };

    /**
     * Wrapper for reading TIFF images from the file system
     * \tparam T_Allocator Allocator(::malloc, ::free) used for managing the raw memory
     */
    template< class T_Allocator = TiffAllocator, ImageFormat T_imgFormat = ImageFormat::ARGB >
    class TiffImage: private boost::noncopyable
    {
        using Allocator = T_Allocator;
        static constexpr ImageFormat imgFormat = T_imgFormat;

        using DataType = typename PixelType<imgFormat>::type;
        using ChannelType = typename PixelType<imgFormat>::ChannelType;
        using Ref = DataType&;
        using ConstRef = const DataType&;

        Allocator alloc_;
        std::string filepath_;
        TIFF* handle_;
        unsigned width_, height_;
        DataType* data_;
        uint16 samplesPerPixel, bitsPerSample, tiffSampleFormat, photometric;

        void
        loadData();

        template< uint16_t T_numChannels, bool T_minIsBlack >
        void
        convert(char* tmp);

    public:

        TiffImage(): TiffImage(Allocator()){}
        TiffImage(const Allocator& alloc): alloc_(alloc), filepath_(""), handle_(nullptr), data_(nullptr){}
        TiffImage(const std::string& filePath): TiffImage(Allocator())
        {
            open(filePath);
        }

        ~TiffImage()
        {
            close();
        }

        void open(const std::string& filePath);
        void close();
        bool isOpen() const
        {
            return (handle_ != nullptr);
        }

        unsigned getWidth() const
        {
            assert(isOpen());
            return width_;
        }

        unsigned getHeight() const
        {
            assert(isOpen());
            return height_;
        }

        size_t getDataSize() const
        {
            assert(isOpen());
            return sizeof(DataType) * width_ * height_;
        }

        Ref
        operator()(unsigned x, unsigned y)
        {
            assert(isOpen());
            return data_[(height_ - 1 - y) * width_ + x];
        }

        ConstRef
        operator()(unsigned x, unsigned y) const
        {
            assert(isOpen());
            return data_[(height_ - 1 - y) * width_ + x];
        }
    };

}  // namespace libTiff

#include "libTiff/libTiff.tpp"
