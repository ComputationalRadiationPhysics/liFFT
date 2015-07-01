#include <cstdlib>
#include <iostream>
#include <fstream>
#include "foobar/libraries/fftw/fftw3Include.h"
#include "foobar/types/AddDimsWrapper.hpp"
#include "foobar/types/DimOffsetWrapper.hpp"
#include <libTiff/traitsAndPolicies.hpp>
#include "policyTest.hpp"
#include "Volume.hpp"
#include "VolumeAdapter.hpp"
#include "generateData.hpp"
#include "IntensityCalculator_Test.hpp"
#include "foobar/FFT.hpp"
#include "foobar/libraries/fftw/FFTW.hpp"
#ifdef WITH_CUDA
#include "foobar/libraries/cuFFT/cuFFT.hpp"
#endif
#include "libTiff/libTiff.hpp"
#include "foobar/policies/ImageAccessor.hpp"
#include "foobar/mem/FileContainer.hpp"
#include "foobar/policies/VolumeAccessor.hpp"
#include "foobar/policies/StreamAccessor.hpp"
#include "foobar/policies/TransposeAccessor.hpp"
#include "foobar/policies/TransformAccessor.hpp"
#include "foobar/types/SymmetricWrapper.hpp"
#include "foobar/FFT_DataWrapper.hpp"
#include <cmath>
#include <type_traits>
#include "foobar/c++14_types.hpp"
#include "testUtils.hpp"

template< typename T = double >
struct MyComplex{
	T real, imag;
	MyComplex(){}
    MyComplex(T real): real(real), imag(0){}
    MyComplex(T real, T imag): real(real), imag(imag){}
};

namespace foobar {
    namespace traits {

        template<typename T>
        struct IsComplex< MyComplex<T> >: std::true_type{};

        template<typename T>
        struct IsBinaryCompatibleImpl< MyComplex<T>, foobar::types::Complex<T> >: std::true_type{};

        template<typename T>
        struct IntegralType< MyComplex<T> >
        {
            using type = T; // or define this in MyComplex itself
        };

        template<>
        struct IsComplex< fftw_complex >: std::true_type{};

    }  // namespace traits

}  // namespace foobar

template<typename T>
std::ostream& operator<< (std::ostream& stream, MyComplex<T> val){
	stream << val.real << " " << val.imag;
	return stream;
}

using ComplexVol     = Volume< MyComplex<> >;
using ComplexVolFFTW = Volume< fftw_complex >;
using RealVol        = Volume<>;

using foobar::types::DimOffsetWrapper;
using ComplexVol2D     = DimOffsetWrapper< ComplexVol,     1 >;
using ComplexVolFFTW2D = DimOffsetWrapper< ComplexVolFFTW, 1 >;
using RealVol2D        = DimOffsetWrapper< RealVol,        1 >;

void testComplex()
{
    ComplexVol2D aperture(1024, 1024);
    ComplexVol2D fftResult(aperture.xDim(), aperture.yDim(), aperture.zDim());
    using FFT_Type = foobar::FFT_2D_C2C_D;
    auto input = FFT_Type::wrapFFT_Input(aperture, foobar::policies::VolumeAccessor());
    auto output = FFT_Type::wrapFFT_Output(fftResult, foobar::policies::VolumeAccessor());
    auto fft = foobar::makeFFT<foobar::libraries::fftw::FFTW<>>(input, output);
	generateData(aperture, Rect<double>(20,20,500,500), foobar::policies::VolumeAccessor());
	fft(input, output);
}

void testReal()
{
    RealVol2D aperture(1024, 1024);
    ComplexVolFFTW2D fftResult(aperture.xDim()/2+1, aperture.yDim(), aperture.zDim());
    RealVol2D intensity(aperture.xDim(), aperture.yDim(), aperture.zDim());
    using FFT_Type = foobar::FFT_2D_R2C_D;
    auto input = FFT_Type::wrapFFT_Input(aperture, foobar::policies::VolumeAccessor());
    auto output = FFT_Type::wrapFFT_Output(fftResult, foobar::policies::VolumeAccessor());
    auto fft = foobar::makeFFT<foobar::libraries::fftw::FFTW<>>(input, output);
    generateData(aperture, Rect<double>(20,20,500,500), foobar::policies::VolumeAccessor());
    fft(input, output);
}

template< typename T_File >
void testFile( T_File& file )
{
    using FFTResult_t = DimOffsetWrapper< Volume< MyComplex<float> >, 1 >;
    FFTResult_t fftResult(file.getExtents()[0]/2+1, file.getExtents()[1]);
    using FFT_Type = foobar::FFT_2D_R2C_F;
    auto input = FFT_Type::wrapFFT_Input(file);
    auto output = FFT_Type::wrapFFT_Output(fftResult, foobar::policies::VolumeAccessor());
#ifdef WITH_CUDA
    using Library = foobar::libraries::cuFFT::CuFFT<>;
#else
    using Library = foobar::libraries::fftw::FFTW<>;
#endif
    auto fft = foobar::makeFFT<Library>(input, output);
    file.loadData(true);
    fft(input, output);
}

/*
 *
 */
int main(int argc, char** argv) {
    //test();
    //testIntensityCalculator();
    //testReal();
    //testComplex();
    initTest();
    visualizeBaseTest();
    using FileType = foobar::mem::FileContainer<
        libTiff::Image<>,
        foobar::policies::ImageAccessorGetColorAsFp<>,
        float,
        false
        >;
    FileType myFile("rect.tif");
    //testFile(myFile);

    finalizeTest();
    return 0;
}

