#include "libTiff.hpp"
#include "libTiff/exceptions.hpp"
#include "libTiff/converters.hpp"

namespace libTiff {

    template< class T_Allocator, ImageFormat T_imgFormat >
    void
    TiffImage< T_Allocator, T_imgFormat >::open(const std::string& filePath)
    {
        handle_ = TIFFOpen(filePath.c_str(), "r");
        filepath_ = filePath;
        uint32 w, h;
        if(!TIFFGetField(handle_, TIFFTAG_IMAGEWIDTH, &w))
            throw InfoMissingException("Width");
        if(!TIFFGetField(handle_, TIFFTAG_IMAGELENGTH, &h))
            throw InfoMissingException("Height");
        width_ = w; height_ = h;
        loadData();
    }

    template< class T_Allocator, ImageFormat T_imgFormat >
    void
    TiffImage< T_Allocator, T_imgFormat >::close()
    {
        if(!handle_)
            return;
        TIFFClose(handle_);
        alloc_.free(data_);
        handle_ = nullptr;
    }

    template< typename T_Data, unsigned T_inStride = 1, unsigned T_outStride = 1 >
    struct ReadTiff{
        static constexpr unsigned inStride = T_inStride;
        static constexpr unsigned outStride = T_outStride;

        TIFF* handle_;
        unsigned width_, height_;
        T_Data* data_;
        char* tmp_;

        ReadTiff(TIFF* handle, unsigned w, unsigned h, T_Data* data, char* tmp):
            handle_(handle), width_(w), height_(h), data_(data), tmp_(tmp){}

        template< typename T >
        void
        assign(T& dst, T&& src)
        {
            dst = std::forward<T>(src);
        }

        template< typename T >
        void
        assign(T& dst, std::array<T, 4>&& src)
        {
            *reinterpret_cast<std::array<T,4>*>(&dst) = std::forward<std::array<T, 4>>(src);
        }

        template<typename T_Func>
        void operator()(T_Func func) {
            using SrcType = typename T_Func::Src;
            SrcType* tmp2 = reinterpret_cast<SrcType*>(tmp_);
            for (unsigned y = 0; y < height_; y++) {
                if(TIFFReadScanline(handle_, tmp_, y) != 1)
                    throw std::runtime_error("Failed reading scanline\n");
                for (unsigned x = 0; x < width_; x++) {
                    assign(data_[y*width_ + x], func(tmp2[x]));
                }
            }
        }
    };

    template< class T_Allocator, ImageFormat T_imgFormat >
    void
    TiffImage< T_Allocator, T_imgFormat >::loadData()
    {
        if(data_)
            return;
        alloc_.malloc(data_, getDataSize());
        if(!data_)
            throw std::runtime_error("Out of memory");
        uint16 samplesPerPixel, bitsPerSample, tiffSampleFormat;
        if(!TIFFGetField(handle_, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel))
            throw InfoMissingException("Samples per pixel");
        if(!TIFFGetField(handle_, TIFFTAG_BITSPERSAMPLE, &bitsPerSample))
            throw InfoMissingException("Bits per sample");
        if(!TIFFGetField(handle_, TIFFTAG_SAMPLEFORMAT, &tiffSampleFormat)){
            std::cerr << "SampelFormat not found. Assuming unsigned";
            tiffSampleFormat = SAMPLEFORMAT_UINT;
        }

        if(needConversion<T_imgFormat>(tiffSampleFormat, samplesPerPixel, bitsPerSample))
        {
            char* tmp;
            size_t numBytes = samplesPerPixel*bitsPerSample/sizeof(char)*width_;
            if(numBytes != TIFFScanlineSize(handle_))
                throw FormatException("Scanline size is unexpected");
            if(samplesPerPixel != 1 && samplesPerPixel != 4)
                throw FormatException("Unsupported sample count");
            alloc_.malloc(tmp, numBytes);
            static constexpr bool dataIsARGB = T_imgFormat == ImageFormat::ARGB;
            if(samplesPerPixel == 1)
            {
                ReadTiff<DataType, dataIsARGB ? 4 : 1, 1 > read(handle_, width_, height_, data_, tmp);
                //Mono pictures
                if(tiffSampleFormat == SAMPLEFORMAT_UINT && bitsPerSample == 8)
                    read(Convert<uint8_t, DataType, false, dataIsARGB>());
                else if(tiffSampleFormat == SAMPLEFORMAT_UINT && bitsPerSample == 16)
                    read(Convert<uint16_t, DataType, false, dataIsARGB>());
                else if(tiffSampleFormat == SAMPLEFORMAT_UINT && bitsPerSample == 32)
                    read(Convert<uint32_t, DataType, false, dataIsARGB>());
                else if(tiffSampleFormat == SAMPLEFORMAT_INT && bitsPerSample == 8)
                    read(Convert<int8_t, DataType, false, dataIsARGB>());
                else if(tiffSampleFormat == SAMPLEFORMAT_INT && bitsPerSample == 16)
                    read(Convert<int16_t, DataType, false, dataIsARGB>());
                else if(tiffSampleFormat == SAMPLEFORMAT_INT && bitsPerSample == 32)
                    read(Convert<int32_t, DataType, false, dataIsARGB>());
                else if(tiffSampleFormat == SAMPLEFORMAT_IEEEFP && bitsPerSample == 32)
                    read(Convert<float, DataType, false, dataIsARGB>());
                else if(tiffSampleFormat == SAMPLEFORMAT_IEEEFP && bitsPerSample == 64)
                    read(Convert<double, DataType, false, dataIsARGB>());
                else
                    throw FormatException("Unimplemented format");
            }else{
                ReadTiff<DataType, dataIsARGB ? 4 : 1, 4 > read(handle_, width_, height_, data_, tmp);
                //Mono pictures
                if(tiffSampleFormat == SAMPLEFORMAT_UINT && bitsPerSample == 8)
                    read(Convert<uint8_t, DataType, true, dataIsARGB>());
                else if(tiffSampleFormat == SAMPLEFORMAT_UINT && bitsPerSample == 16)
                    read(Convert<uint16_t, DataType, true, dataIsARGB>());
                else if(tiffSampleFormat == SAMPLEFORMAT_UINT && bitsPerSample == 32)
                    read(Convert<uint32_t, DataType, true, dataIsARGB>());
                else if(tiffSampleFormat == SAMPLEFORMAT_INT && bitsPerSample == 8)
                    read(Convert<int8_t, DataType, true, dataIsARGB>());
                else if(tiffSampleFormat == SAMPLEFORMAT_INT && bitsPerSample == 16)
                    read(Convert<int16_t, DataType, true, dataIsARGB>());
                else if(tiffSampleFormat == SAMPLEFORMAT_INT && bitsPerSample == 32)
                    read(Convert<int32_t, DataType, true, dataIsARGB>());
                else if(tiffSampleFormat == SAMPLEFORMAT_IEEEFP && bitsPerSample == 32)
                    read(Convert<float, DataType, true, dataIsARGB>());
                else if(tiffSampleFormat == SAMPLEFORMAT_IEEEFP && bitsPerSample == 64)
                    read(Convert<double, DataType, true, dataIsARGB>());
                else
                    throw FormatException("Unimplemented format");
            }
            alloc_.free(tmp);
        }else{
            if(getDataSize() != TIFFScanlineSize(handle_)*height_)
                throw FormatException("Scanline size is unexpected");
            for (unsigned y = 0; y < height_; y++) {
                if(TIFFReadScanline(handle_, &data_[y*width_], y) != 1)
                    throw std::runtime_error("Failed reading scanline\n");
            }
        }
    }

}  // namespace libTiff
