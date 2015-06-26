#include "libTiff.hpp"
#include "libTiff/exceptions.hpp"
#include "libTiff/converters.hpp"

namespace libTiff {

    template< ImageFormat T_imgFormat, class T_Allocator >
    void
    Image< T_imgFormat, T_Allocator >::openHandle(const std::string& filePath, const char* mode)
    {
        closeHandle();
        handle_ = TIFFOpen(filePath.c_str(), mode);
        if(!handle_)
            throw std::runtime_error("Could not open "+filePath);
        filepath_ = filePath;
    }

    template< ImageFormat T_imgFormat, class T_Allocator >
    void
    Image< T_imgFormat, T_Allocator >::closeHandle()
    {
        if(handle_){
            TIFFClose(handle_);
            handle_ = nullptr;
            isReadable_ = false;
            isWriteable_ = false;
            dataWritten_ = false;
        }
    }

    template< ImageFormat T_imgFormat, class T_Allocator >
    void
    Image< T_imgFormat, T_Allocator >::open(const std::string& filePath, bool loadData)
    {
        close();
        openHandle(filePath, "r");
        isReadable_ = true;
        uint32 w, h;
        if(!TIFFGetField(handle_, TIFFTAG_IMAGEWIDTH, &w))
            throw InfoMissingException("Width");
        if(!TIFFGetField(handle_, TIFFTAG_IMAGELENGTH, &h))
            throw InfoMissingException("Height");
        width_ = w; height_ = h;
        if(loadData)
            this->loadData();
        else
            allocData();
    }

    template< ImageFormat T_imgFormat, class T_Allocator >
    void
    Image< T_imgFormat, T_Allocator >::open(const std::string& filePath, unsigned w, unsigned h)
    {
        close();
        openHandle(filePath, "w");
        isWriteable_ = true;
        width_ = w; height_ = h;
        allocData();
    }

    template< ImageFormat T_imgFormat, class T_Allocator >
    void
    Image< T_imgFormat, T_Allocator >::close()
    {
        closeHandle();
        alloc_.free(data_);
        data_ = nullptr;
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
                if(!TIFFReadScanline(handle_, tmp_, y))
                    throw std::runtime_error("Failed reading scanline\n");
                for (unsigned x = 0; x < width_; x++) {
                    assign(data_[(y*width_ + x)*outStride], func(srcChannels[x*inStride]));
                }
            }
        }
    };

    template< ImageFormat T_imgFormat, class T_Allocator >
    template< uint16_t T_numChannels, bool T_minIsBlack >
    void
    Image< T_imgFormat, T_Allocator >::convert(char* tmp)
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

    template< ImageFormat T_imgFormat, class T_Allocator >
    void
    Image< T_imgFormat, T_Allocator >::allocData()
    {
        if(data_)
            return;
        alloc_.malloc(data_, getDataSize());
        if(!data_)
            throw std::runtime_error("Out of memory");
    }

    template< ImageFormat T_imgFormat, class T_Allocator >
    void
    Image< T_imgFormat, T_Allocator >::load()
    {
        if(!isReadable_)
            throw std::runtime_error("Cannot load file that is not opened for reading");
        loadData();
    }

    template< ImageFormat T_imgFormat, class T_Allocator >
    template< typename T>
    void
    Image< T_imgFormat, T_Allocator >::checkedWrite(uint16 tag, T value){
        if(!TIFFSetField(handle_, tag, value))
            throw InfoWriteException(std::to_string(tag));
    }


    template< ImageFormat T_imgFormat, class T_Allocator >
    void
    Image< T_imgFormat, T_Allocator >::saveTo(const std::string& filePath, bool compress, bool saveAsARGB)
    {
        openHandle(filePath, "w");
        isWriteable_ = true;
        save(compress, saveAsARGB);
    }

    template< ImageFormat T_imgFormat, class T_Allocator >
    void
    Image< T_imgFormat, T_Allocator >::save(bool compress, bool saveAsARGB)
    {
        if(!isWriteable_)
            throw std::runtime_error("Cannot save to a file that is not opened for writing");
        // If we already wrote to the image we need to reopen it to be able to modify the tags
        if(dataWritten_){
            openHandle(filepath_, "w");
            isWriteable_ = true;
        }
        if(imgFormat == ImageFormat::ARGB && !saveAsARGB)
            checkedWrite(TIFFTAG_SAMPLESPERPIXEL, 3); // Write as RGB
        else
            checkedWrite(TIFFTAG_SAMPLESPERPIXEL, SamplesPerPixel<imgFormat>::value);
        checkedWrite(TIFFTAG_BITSPERSAMPLE, BitsPerSample<imgFormat>::value);
        checkedWrite(TIFFTAG_SAMPLEFORMAT, PixelType<imgFormat>::tiffType);
        checkedWrite(TIFFTAG_IMAGEWIDTH, width_);
        checkedWrite(TIFFTAG_IMAGELENGTH, height_);
        if(imgFormat == ImageFormat::ARGB)
            checkedWrite(TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
        else
            checkedWrite(TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        checkedWrite(TIFFTAG_ROWSPERSTRIP, 1);
        checkedWrite(TIFFTAG_XRESOLUTION, 1.);
        checkedWrite(TIFFTAG_YRESOLUTION, 1.);
        checkedWrite(TIFFTAG_RESOLUTIONUNIT, 1);
        if(compress)
            checkedWrite(TIFFTAG_COMPRESSION, COMPRESSION_LZW);
        else
            checkedWrite(TIFFTAG_COMPRESSION, COMPRESSION_NONE);
        checkedWrite(TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        checkedWrite(TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
        if(imgFormat == ImageFormat::ARGB && !saveAsARGB){
            // Convert from ARGB to RGB
            using ChannelType = typename PixelType<imgFormat>::ChannelType;
            if(width_ * sizeof(ChannelType) * 3 != TIFFScanlineSize(handle_))
                throw FormatException("Scanline size is unexpected");
            ChannelType* tmpLine;
            alloc_.malloc(tmpLine, TIFFScanlineSize(handle_));
            for (unsigned y = 0; y < height_; y++)
            {
                auto* lineData = &data_[y*width_];
                for(unsigned x = 0; x < width_; x++)
                {
                    tmpLine[x*3] = static_cast<ChannelType>(TIFFGetR(lineData[x]));
                    tmpLine[x*3+1] = static_cast<ChannelType>(TIFFGetG(lineData[x]));
                    tmpLine[x*3+2] = static_cast<ChannelType>(TIFFGetB(lineData[x]));
                }
                if(!TIFFWriteScanline(handle_, tmpLine, y))
                    throw std::runtime_error("Failed writing scanline");
            }
            alloc_.free(tmpLine);
        }else{
            if(getDataSize() != TIFFScanlineSize(handle_)*height_)
                throw FormatException("Scanline size is unexpected");
            for (unsigned y = 0; y < height_; y++)
            {
                if(!TIFFWriteScanline(handle_, &data_[y*width_], y))
                    throw std::runtime_error("Failed writing scanline");
            }
        }
        dataWritten_ = true;
    }

    template< ImageFormat T_imgFormat, class T_Allocator >
    void
    Image< T_imgFormat, T_Allocator >::loadData()
    {
        allocData();
        if(!TIFFGetField(handle_, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel))
            throw InfoMissingException("Samples per pixel");
        if(!TIFFGetField(handle_, TIFFTAG_BITSPERSAMPLE, &bitsPerSample))
            throw InfoMissingException("Bits per sample");
        if(!TIFFGetField(handle_, TIFFTAG_SAMPLEFORMAT, &tiffSampleFormat)){
            std::cerr << "SampelFormat not found. Assuming unsigned" << std::endl;
            tiffSampleFormat = SAMPLEFORMAT_UINT;
        }
        uint16 planarConfig;
        if(!TIFFGetField(handle_, TIFFTAG_PLANARCONFIG, &planarConfig) || planarConfig!=PLANARCONFIG_CONTIG){
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
            if(numBytes != static_cast<size_t>(TIFFScanlineSize(handle_)))
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
            if(getDataSize() != static_cast<size_t>(TIFFScanlineSize(handle_))*height_)
                throw FormatException("Scanline size is unexpected");
            for (unsigned y = 0; y < height_; y++) {
                if(TIFFReadScanline(handle_, &data_[y*width_], y) != 1)
                    throw std::runtime_error("Failed reading scanline");
            }
        }
    }

}  // namespace libTiff
