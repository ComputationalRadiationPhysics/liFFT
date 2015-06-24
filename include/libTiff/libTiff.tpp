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

    template< typename T_Data, uint16_t T_inStride = 1, uint16_t T_outStride = 1 >
    struct ReadTiff{
        static constexpr uint16_t inStride = T_inStride;
        static constexpr uint16_t outStride = T_outStride;

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

        template< typename T, size_t T_numChannels >
        void
        assign(T& dst, std::array< T, T_numChannels >&& src)
        {
            *reinterpret_cast<std::array< T, T_numChannels >*>(&dst) = std::forward<std::array< T, T_numChannels >>(src);
        }

        template<typename T_Func>
        void operator()(T_Func func) {
            using SrcType = typename T_Func::Src;
            SrcType* srcChannels = reinterpret_cast<SrcType*>(tmp_);
            for (unsigned y = 0; y < height_; y++) {
                if(TIFFReadScanline(handle_, tmp_, y) != 1)
                    throw std::runtime_error("Failed reading scanline\n");
                for (unsigned x = 0; x < width_; x++) {
                    assign(data_[(y*width_ + x)*outStride], func(srcChannels[x*inStride]));
                }
            }
        }
    };

    template< class T_Allocator, ImageFormat T_imgFormat >
    template< uint16_t T_numChannels, bool T_minIsBlack >
    void
    TiffImage< T_Allocator, T_imgFormat >::convert(char* tmp)
    {
        static constexpr uint16_t numChannelsSrc = T_numChannels;
        static constexpr uint16_t numChannelsDest = SamplesPerPixel<T_imgFormat>::value;
        static constexpr bool minIsBlack = T_minIsBlack;
        ReadTiff<ChannelType, numChannelsSrc, numChannelsDest > read(handle_, width_, height_, reinterpret_cast<ChannelType*>(data_), tmp);
        //Mono pictures
        if(tiffSampleFormat == SAMPLEFORMAT_UINT && bitsPerSample == 8)
            read(Convert<uint8_t, ChannelType, numChannelsSrc, numChannelsDest, minIsBlack>());
        else if(tiffSampleFormat == SAMPLEFORMAT_UINT && bitsPerSample == 16)
            read(Convert<uint16_t, ChannelType, numChannelsSrc, numChannelsDest, minIsBlack>());
        else if(tiffSampleFormat == SAMPLEFORMAT_UINT && bitsPerSample == 32)
            read(Convert<uint32_t, ChannelType, numChannelsSrc, numChannelsDest, minIsBlack>());
        else if(tiffSampleFormat == SAMPLEFORMAT_INT && bitsPerSample == 8)
            read(Convert<int8_t, ChannelType, numChannelsSrc, numChannelsDest, minIsBlack>());
        else if(tiffSampleFormat == SAMPLEFORMAT_INT && bitsPerSample == 16)
            read(Convert<int16_t, ChannelType, numChannelsSrc, numChannelsDest, minIsBlack>());
        else if(tiffSampleFormat == SAMPLEFORMAT_INT && bitsPerSample == 32)
            read(Convert<int32_t, ChannelType, numChannelsSrc, numChannelsDest, minIsBlack>());
        else if(tiffSampleFormat == SAMPLEFORMAT_IEEEFP && bitsPerSample == 32)
            read(Convert<float, ChannelType, numChannelsSrc, numChannelsDest, minIsBlack>());
        else if(tiffSampleFormat == SAMPLEFORMAT_IEEEFP && bitsPerSample == 64)
            read(Convert<double, ChannelType, numChannelsSrc, numChannelsDest, minIsBlack>());
        else
            throw FormatException("Unimplemented format");
    }

    template< class T_Allocator, ImageFormat T_imgFormat >
    void
    TiffImage< T_Allocator, T_imgFormat >::loadData()
    {
        if(data_)
            return;
        alloc_.malloc(data_, getDataSize());
        if(!data_)
            throw std::runtime_error("Out of memory");
        if(!TIFFGetField(handle_, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel))
            throw InfoMissingException("Samples per pixel");
        if(!TIFFGetField(handle_, TIFFTAG_BITSPERSAMPLE, &bitsPerSample))
            throw InfoMissingException("Bits per sample");
        if(!TIFFGetField(handle_, TIFFTAG_SAMPLEFORMAT, &tiffSampleFormat)){
            std::cerr << "SampelFormat not found. Assuming unsigned" << std::endl;
            tiffSampleFormat = SAMPLEFORMAT_UINT;
        }
        uint16 planarConfig;
        if(!TIFFGetField(handle_, TIFFTAG_PLANARCONFIG, &planarConfig) || planarConfig!=1){
            throw FormatException("PlanarConfig missing or not 1");
        }
        if(!TIFFGetField(handle_, TIFFTAG_PHOTOMETRIC, &photometric))
            throw InfoMissingException("Photometric");
        if(photometric != PHOTOMETRIC_RGB && photometric != PHOTOMETRIC_MINISBLACK && photometric != PHOTOMETRIC_MINISWHITE)
            throw FormatException("Photometric is not supported: " + std::to_string(photometric));

        if(needConversion<T_imgFormat>(tiffSampleFormat, samplesPerPixel, bitsPerSample))
        {
            char* tmp;
            size_t numBytes = samplesPerPixel*bitsPerSample/8*width_;
            if(numBytes != TIFFScanlineSize(handle_))
                throw FormatException("Scanline size is unexpected: "+std::to_string(numBytes)+":"+std::to_string(TIFFScanlineSize(handle_)));
            if(samplesPerPixel != 1 && samplesPerPixel != 3 && samplesPerPixel != 4)
                throw FormatException("Unsupported sample count");
            alloc_.malloc(tmp, numBytes);
            if(samplesPerPixel == 1)
            {
                if(photometric == PHOTOMETRIC_MINISWHITE)
                    convert<1, false>(tmp);
                else
                    convert<1, true>(tmp);
            }else if(samplesPerPixel == 3){
                if(photometric == PHOTOMETRIC_MINISWHITE)
                    convert<3, false>(tmp);
                else
                    convert<3, true>(tmp);
            }else{
                if(photometric == PHOTOMETRIC_MINISWHITE)
                    convert<4, false>(tmp);
                else
                    convert<4, true>(tmp);
            }
            alloc_.free(tmp);
        }else{
            if(getDataSize() != TIFFScanlineSize(handle_)*height_)
                throw FormatException("Scanline size is unexpected");
            for (unsigned y = 0; y < height_; y++) {
                if(TIFFReadScanline(handle_, &data_[y*width_], y) != 1)
                    throw std::runtime_error("Failed reading scanline");
            }
        }
    }

}  // namespace libTiff
