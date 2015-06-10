#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>
#include <fftw3.h>
#include <libTiff/traitsAndPolicies.hpp>
#include "c++14_types.hpp"
#include "policyTest.hpp"
#include "Volume.hpp"
#include "VolumeAdapter.hpp"
#include "generateData.hpp"
#include "IntensityCalculator_Test.hpp"
#include "foobar/policies/FFT.hpp"
#include "foobar/libraries/fftw/FFTW.hpp"
#include "foobar/libraries/cuFFT/cuFFT.hpp"
#include "libTiff/libTiff.hpp"
#include "foobar/policies/ImageAccessor.hpp"
#include "foobar/types/FileContainer.hpp"
#include "foobar/policies/DataContainerAccessor.hpp"
#include "foobar/policies/VolumeAccessor.hpp"

template< typename T = double >
struct MyComplex{
	T real, imag;
	MyComplex(){}
	MyComplex(T real): real(real), imag(0){}
};

namespace foobar {
    namespace traits {

        template<typename T>
        struct IsComplex< MyComplex<T> >: std::true_type{};

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

template<typename T> inline
T absSqr(const MyComplex<T>& val){
	return val.real*val.real + val.imag*val.imag;
}

double absSqr(const fftw_complex& val){
	return val[0]*val[0] + val[1]*val[1];
}

template< typename T, class T_Accessor >
void write2File(T& data, T_Accessor&& acc, const std::string& name){
    std::ofstream inFile(name.c_str());
    foobar::policies::GetExtents<T> extents(data);
    foobar::types::Vec<2> idx;
    for(idx[0]=0; idx[0]<extents[0]; ++idx[0]){
        for(idx[1]=0; idx[1]<extents[1]; ++idx[1]){
            inFile << acc(idx, data);
            if(idx[1]+1 < extents[1])
                inFile << " ";
        }
        if(idx[0]+1 < extents[0])
            inFile << "\n";
    }
    inFile.close();
}

template<typename T, typename T2>
void calcIntensities(const T& in, T2& out, size_t numX, size_t numY){
    for(size_t y=0; y<numY; ++y){
        for(size_t x=0; x<numX; ++x){
            out(x,y) = absSqr(in(x,y));
        }
    }
}

using ComplexVol = Volume< MyComplex<> >;
using ComplexVolFFTW = Volume< fftw_complex >;
using RealVol = Volume<>;

void testComplex()
{
    ComplexVol aperture(1024, 1024);
    ComplexVol fftResult(aperture.xDim(), aperture.yDim(), aperture.zDim());
	RealVol intensity(aperture.xDim(), aperture.yDim(), aperture.zDim());
    //fftw_plan plan = fftw_plan_dft_2d(aperture.yDim(), aperture.xDim(), reinterpret_cast<fftw_complex*>(aperture.data()), reinterpret_cast<fftw_complex*>(fftResult.data()), FFTW_FORWARD, FFTW_ESTIMATE);
    using FFTType = typename foobar::policies::FFT<
                        foobar::libraries::cuFFT::CuFFT<>,
                        ComplexVol,
                        ComplexVol,
                        foobar::AutoDetect,
                        std::integral_constant<unsigned, 2>
                    >::type;
    FFTType fft(aperture, fftResult);
	generateData(aperture, Rect<double>(20,20));
	//fftw_execute(plan);
	//fftw_destroy_plan(plan);
	fft(aperture, fftResult);
	calcIntensities(aperture, intensity, aperture.xDim(), aperture.yDim());
	write2File(intensity, foobar::policies::VolumeAccessor(), "input.txt");
	calcIntensities(fftResult, intensity, fftResult.xDim(), fftResult.yDim());
	auto adapter = makeTransposeAdapter(intensity);
	write2File(adapter, foobar::policies::VolumeAccessor(), "output.txt");
}

void testReal()
{
    RealVol aperture(1024, 1024);
    ComplexVolFFTW fftResult(aperture.xDim()/2+1, aperture.yDim(), aperture.zDim());
    RealVol intensity(aperture.xDim(), aperture.yDim(), aperture.zDim());
    //fftw_plan plan = fftw_plan_dft_r2c_2d(aperture.yDim(), aperture.xDim(), aperture.data(), reinterpret_cast<fftw_complex*>(fftResult.data()), FFTW_ESTIMATE);
    using FFTType = typename foobar::policies::FFT<
                        foobar::libraries::fftw::FFTW<>,
                        RealVol, ComplexVolFFTW,
                        foobar::AutoDetect,
                        std::integral_constant<unsigned, 2>
                    >::type;
    FFTType fft(aperture, fftResult);
    generateData(aperture, Rect<double>(20,20));
    write2File(aperture, foobar::policies::VolumeAccessor(), "input.txt");
    //fftw_execute(plan);
    //fftw_destroy_plan(plan);
    fft(aperture, fftResult);
    SymetricAdapter<fftw_complex> symAdapter(aperture.xDim(), fftResult);
    calcIntensities(symAdapter, intensity, symAdapter.xDim(), symAdapter.yDim());
    auto adapter = makeTransposeAdapter(intensity);
    write2File(adapter, foobar::policies::VolumeAccessor(), "output.txt");
}

template< typename T_File >
void testFile( T_File& file )
{
    using FFTResult_t = Volume< MyComplex<float> >;
    FFTResult_t fftResult(file.getExtents()[0]/2+1, file.getExtents()[1]);
    Volume< float > intensity(file.getExtents()[0], file.getExtents()[1]);
    //fftw_plan plan = fftw_plan_dft_r2c_2d(aperture.yDim(), aperture.xDim(), aperture.data(), reinterpret_cast<fftw_complex*>(fftResult.data()), FFTW_ESTIMATE);
    using FFTType = typename foobar::policies::FFT<
                        foobar::libraries::fftw::FFTW<>,
                        T_File,
                        FFTResult_t,
                        foobar::AutoDetect,
                        std::integral_constant<unsigned, 2>
                    >::type;
    FFTType fft(file, fftResult);
    file.loadData(true);
    fft(file, fftResult);
    SymetricAdapter< MyComplex<float> > symAdapter(intensity.xDim(), fftResult);
    calcIntensities(symAdapter, intensity, symAdapter.xDim(), symAdapter.yDim());
    auto adapter = makeTransposeAdapter(intensity);
    write2File(file.getData(), foobar::policies::DataContainerAccessor(), "input.txt");
    write2File(adapter, foobar::policies::VolumeAccessor(), "output.txt");
}

/*
 *
 */
int main(int argc, char** argv) {
    //test();
    //testIntensityCalculator();
    //testComplex();
    using FileType = foobar::types::FileContainer<
        libTiff::TiffImage<>,
        foobar::policies::ImageAccessorGetColorAsFp<>,
        foobar::policies::DataContainerAccessor,
        float,
        false
        >;
    FileType myFile("/home/grund59/Rect.tif");
    testFile(myFile);
    //testReal();
    if(std::system("python writeData.py -i input.txt -o input.pdf"));
    if(std::system("python writeData.py -s -i output.txt -o output.pdf"));
    return 0;
}

