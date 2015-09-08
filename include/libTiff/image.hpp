#pragma once

#include <string>
#include <cassert>
#include <tiffio.h>
#include <boost/utility.hpp>
#include "libTiff/ImageFormat.hpp"
#include "libTiff/FormatTraits.hpp"
#include <memory>

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
    template< ImageFormat T_imgFormat = ImageFormat::ARGB, class T_Allocator = TiffAllocator >
    class Image
    {
        using Allocator = T_Allocator;
        static constexpr ImageFormat imgFormat = T_imgFormat;

        using DataType = typename PixelType<imgFormat>::type;
        using ChannelType = typename PixelType<imgFormat>::ChannelType;
        using Ref = DataType&;
        using ConstRef = const DataType&;

        std::string filepath_;
        std::unique_ptr<TIFF, void(*)(TIFF*)> handle_;
        std::unique_ptr<DataType[], void(*)(DataType*)> data_;
        bool isReadable_, isWriteable_, dataWritten_;
        unsigned width_, height_;
        uint16 samplesPerPixel, bitsPerSample, tiffSampleFormat, photometric;

        void openHandle(const std::string& filePath, const char* mode);
        void closeHandle();
        void allocData();
        void loadData();
        template<typename T>
        void checkedWrite(uint16 tag, T value);

        template< uint16_t T_numChannels, bool T_minIsBlack >
        void
        convert(char* tmp);

        Image(const Image&) = delete;
        Image& operator=(const Image&) = delete;
    public:

        /**
         * Creates an invalid image using the standard allocator.
         * Before accessing it you need to call \ref open(..)
         */
        Image():
            filepath_(""),
            handle_(nullptr, TIFFClose),
            data_(nullptr, [](DataType* p){ Allocator().free(p);}),
            isReadable_(false),
            isWriteable_(false),
            dataWritten_(false),
            width_(0), height_(0)
        {}
        Image(Image&&) = default;
        Image& operator=(Image&&) = default;

        /**
         * Opens the image at the given filePath for reading
         *
         * @param filePath Path to the image to load
         * @param loadData True if the image data should be loaded or only its memory allocated.
         *          The data can be (re)loaded with \ref load()
         */
        Image(const std::string& filePath, bool loadData = true): Image()
        {
            open(filePath, loadData);
        }

        /**
         * Opens the image at the given filePath for writing
         * Overwrites or creates it
         *
         * @param filePath Path to the image to save to
         * @param w Width of the new image
         * @param h Height of the new image
         */
        Image(const std::string& filePath, unsigned w, unsigned h): Image()
        {
            open(filePath, w, h);
        }

        ~Image()
        {
            close();
        }

        /**
         * Opens the image at the given filePath for reading
         * Implicitly closes an open image
         *
         * @param filePath Path to the image to load
         * @param loadData True if the image data should be loaded or only its memory allocated.
         *          The data can be (re)loaded with \ref load()
         */
        void open(const std::string& filePath, bool loadData = true);

        /**
         * Opens the image at the given filePath for writing
         * Overwrites or creates it
         * Implicitly closes an open image
         *
         * @param filePath Path to the image to save to
         * @param w Width of the new image
         * @param h Height of the new image
         */
        void open(const std::string& filePath, unsigned w, unsigned h);

        /**
         * Closes the current image freeing all memory
         */
        void close();

        /**
         * Flushes the current write buffer to file invalidating the file handle
         * You have to use saveTo afterwards!
         */
        void flush(){ closeHandle(); }

        /**
         * Loads the image data into memory
         * Throws an exception if the image is not opened for reading
         */
        void load();

        /**
         * Saves the image data to file. This is NOT done in the destructor!
         * Throws an exception if the image is not opened for writing
         *
         * @param compress Whether to compress the file or not
         * @param saveAsARGB Whether to save ARGB files as ARGB (true) or RGB only (ignored for monochromatic files)
         */
        void save(bool compress = true, bool saveAsARGB = true);

        /**
         * Saves the image data to file at the given path and opens it for writing
         * Can be used to write modified data to a file (might be the same as the current one)
         *
         * @param filePath Path to the image to save to
         * @param compress Whether to compress the file or not
         * @param saveAsARGB Whether to save ARGB files as ARGB (true) or RGB only (ignored for monochromatic files)
         */
        void saveTo(const std::string& filePath, bool compress = true, bool saveAsARGB = true);

        /**
         * Returns whether the image is open and its data can be read
         * @return True if the image is open
         */
        bool isOpen() const
        {
            return (handle_ != nullptr);
        }

        unsigned getWidth() const
        {
            assert(isOpen() || data_);
            return width_;
        }

        unsigned getHeight() const
        {
            assert(isOpen() || data_);
            return height_;
        }

        /**
         * Returns the total size of the used memory for the image data
         * @return size in bytes
         */
        size_t getDataSize() const
        {
            assert(isOpen() || data_);
            return sizeof(DataType) * width_ * height_;
        }

        /**
         * Accesses the pixel at the given location (read-write)
         * @param x
         * @param y
         * @return Reference to the pixel value
         */
        Ref
        operator()(unsigned x, unsigned y)
        {
            assert(isOpen() || data_);
            return data_[(height_ - 1 - y) * width_ + x];
        }

        /**
         * Accesses the pixel at the given location (read-only)
         * @param x
         * @param y
         * @return The pixel value
         */
        ConstRef
        operator()(unsigned x, unsigned y) const
        {
            assert(isOpen() || data_);
            return data_[(height_ - 1 - y) * width_ + x];
        }
    };

    /**
     * Monochrome image where each pixel is represented by 1 float value
     * (A)RGB images will be converted and scaled to [0,1]
     */
    template< class T_Allocator = TiffAllocator >
    using FloatImage = Image< ImageFormat::Float, T_Allocator >;

    /**
     * Monochrome image where each pixel is represented by 1 double value
     * (A)RGB images will be converted and scaled to [0,1]
     */
    template< class T_Allocator = TiffAllocator >
    using DoubleImage = Image< ImageFormat::Double, T_Allocator >;

}  // namespace libTiff

#include "libTiff/image.tpp"
