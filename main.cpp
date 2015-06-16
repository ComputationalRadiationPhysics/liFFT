#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>
#include <fftw3.h>
#include <libTiff/traitsAndPolicies.hpp>
#include "foobar/c++14_types.hpp"
#include "policyTest.hpp"
#include "Volume.hpp"
#include "VolumeAdapter.hpp"
#include "generateData.hpp"
#include "IntensityCalculator_Test.hpp"
#include "foobar/FFT.hpp"
#include "foobar/libraries/fftw/FFTW.hpp"
#include "foobar/libraries/cuFFT/cuFFT.hpp"
#include "libTiff/libTiff.hpp"
#include "foobar/policies/ImageAccessor.hpp"
#include "foobar/types/FileContainer.hpp"
#include "foobar/policies/VolumeAccessor.hpp"
#include "foobar/policies/StreamAccessor.hpp"
#include "foobar/policies/TransposeAccessor.hpp"
#include "foobar/policies/TransformAccessor.hpp"
#include "foobar/types/StreamWrapper.hpp"
#include "foobar/types/DimOffsetWrapper.hpp"
#include "foobar/types/SymmetricWrapper.hpp"
#include "foobar/FFT_DataWrapper.hpp"

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

template< class T_Accessor, typename T >
void write2File(T& data, const std::string& name){
    using F = foobar::policies::Copy< T_Accessor, foobar::policies::StringStreamAccessor<> >;

    foobar::types::StreamWrapper< std::ofstream, 2 > inFile(name.c_str());
    static_assert(foobar::traits::IsStreamAccessor< foobar::policies::StringStreamAccessor<>, decltype(inFile)>::value, "bbb");
    F()(data, inFile);
//    foobar::policies::GetExtents< T, 2 > extents(data);
//    foobar::types::Vec<2> idx;
//    for(idx[0]=0; idx[0]<extents[0]; ++idx[0]){
//        for(idx[1]=0; idx[1]<extents[1]; ++idx[1]){
//            inFile << acc(idx, data);
//            if(idx[1]+1 < extents[1])
//                inFile << " ";
//        }
//        if(idx[0]+1 < extents[0])
//            inFile << "\n";
//    }
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

template< typename T >
struct CalcIntensityFunc
{
    auto
    operator()(const T& val) const
    -> decltype(val.real*val.real + val.imag*val.imag)
    {
        return val.real*val.real + val.imag*val.imag;
    }
};

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
    RealVol2D intensity(aperture.xDim(), aperture.yDim(), aperture.zDim());
    //fftw_plan plan = fftw_plan_dft_2d(aperture.yDim(), aperture.xDim(), reinterpret_cast<fftw_complex*>(aperture.data()), reinterpret_cast<fftw_complex*>(fftResult.data()), FFTW_FORWARD, FFTW_ESTIMATE);
    using FFTType = typename foobar::FFT<
                        foobar::libraries::cuFFT::CuFFT<>,
                        ComplexVol2D,
                        ComplexVol2D
                    >::type;
    FFTType fft(aperture, fftResult);
	generateData(aperture, Rect<double>(20,20));
	//fftw_execute(plan);
	//fftw_destroy_plan(plan);
	fft(aperture, fftResult);
	calcIntensities(aperture, intensity, aperture.xDim(), aperture.yDim());
	write2File<foobar::policies::VolumeAccessor>(intensity, "input.txt");
	calcIntensities(fftResult, intensity, fftResult.xDim(), fftResult.yDim());
	auto adapter = makeTransposeAdapter(intensity);
	write2File<foobar::policies::VolumeAccessor>(adapter, "output.txt");
}

void testReal()
{
    RealVol2D aperture(1024, 1024);
    ComplexVolFFTW2D fftResult(aperture.xDim()/2+1, aperture.yDim(), aperture.zDim());
    RealVol2D intensity(aperture.xDim(), aperture.yDim(), aperture.zDim());
    //fftw_plan plan = fftw_plan_dft_r2c_2d(aperture.yDim(), aperture.xDim(), aperture.data(), reinterpret_cast<fftw_complex*>(fftResult.data()), FFTW_ESTIMATE);
    using FFTType = typename foobar::FFT<
                        foobar::libraries::fftw::FFTW<>,
                        RealVol,
                        ComplexVolFFTW
                    >::type;
    FFTType fft(aperture, fftResult);
    generateData(aperture, Rect<double>(20,20));
    write2File<foobar::policies::VolumeAccessor>(aperture, "input.txt");
    //fftw_execute(plan);
    //fftw_destroy_plan(plan);
    fft(aperture, fftResult);
    SymetricAdapter<fftw_complex> symAdapter(aperture.xDim(), fftResult);
    calcIntensities(symAdapter, intensity, symAdapter.xDim(), symAdapter.yDim());
    auto adapter = makeTransposeAdapter(intensity);
    write2File<foobar::policies::VolumeAccessor>(adapter, "output.txt");
}

template< typename T_File >
void testFile( T_File& file )
{
    using FFTResult_t = DimOffsetWrapper< Volume< MyComplex<float> >, 1 >;
    FFTResult_t fftResult(file.getExtents()[0]/2+1, file.getExtents()[1]);
    foobar::FFT_DataWrapper<FFTResult_t> myFFTResult(fftResult);
    DimOffsetWrapper< Volume< float >, 1 > intensity(file.getExtents()[0], file.getExtents()[1]);
    //fftw_plan plan = fftw_plan_dft_r2c_2d(aperture.yDim(), aperture.xDim(), aperture.data(), reinterpret_cast<fftw_complex*>(fftResult.data()), FFTW_ESTIMATE);
    using FFTType = typename foobar::FFT<
                        foobar::libraries::fftw::FFTW<>,
                        T_File,
                        FFTResult_t
                    >::type;
    FFTType fft(file, fftResult);
    file.loadData(true);
    fft(file, fftResult);
    foobar::types::SymmetricWrapper< decltype(fftResult), foobar::policies::VolumeAccessor > fullResult(fftResult, file.getExtents()[0]);
    using GetIntensityOfOutput =
        foobar::policies::TransformAccessor<
            foobar::policies::TransposeAccessor<>,
            CalcIntensityFunc<
                MyComplex<float>
            >
        >;
    write2File<foobar::policies::DataContainerAccessor>(file.getData(), "input.txt");
    write2File<GetIntensityOfOutput>(fullResult, "output.txt");
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
        float,
        false
        >;
    testReal();
    FileType myFile("/home/grund59/Rect.tif");
    testFile(myFile);
    if(std::system("python writeData.py -i input.txt -o input.pdf"))
        std::cout << "Error converting input\n";
    if(std::system("python writeData.py -s -i output.txt -o output.pdf"))
        std::cout << "Error converting output\n";
    return 0;
}

