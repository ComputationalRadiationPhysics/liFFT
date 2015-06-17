#include "libTiff.hpp"
#include <stdexcept>

namespace libTiff {

    template< class T_Allocator >
    void
    TiffImage<T_Allocator>::open(const std::string& filePath)
    {
        handle_ = TIFFOpen(filePath.c_str(), "r");
        filepath_ = filePath;
        uint32 w, h;
        TIFFGetField(handle_, TIFFTAG_IMAGEWIDTH, &w);
        TIFFGetField(handle_, TIFFTAG_IMAGELENGTH, &h);
        width_ = w; height_ = h;
        loadData();
    }

    template< class T_Allocator >
    void
    TiffImage<T_Allocator>::close()
    {
        if(!handle_)
            return;
        TIFFClose(handle_);
        alloc_.free(data_);
        handle_ = nullptr;
    }

    template< class T_Allocator >
    void
    TiffImage<T_Allocator>::loadData()
    {
        if(data_)
            return;
        alloc_.malloc( data_, getDataSize() );
        if(!data_)
            throw std::runtime_error("Out of memory");
        if(!TIFFReadRGBAImage(handle_, width_, height_, data_, 1))
            throw std::runtime_error("Error reading image");
    }

}  // namespace libTiff
