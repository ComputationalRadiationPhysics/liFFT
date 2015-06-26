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
#include "foobar/types/FileContainer.hpp"
#include "foobar/policies/VolumeAccessor.hpp"
#include "foobar/policies/StreamAccessor.hpp"
#include "foobar/policies/TransposeAccessor.hpp"
#include "foobar/policies/TransformAccessor.hpp"
#include "foobar/types/SymmetricWrapper.hpp"
#include "foobar/FFT_DataWrapper.hpp"
#include <cmath>
#include <type_traits>
#include "foobar/c++14_types.hpp"

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

template< class T_Accessor, typename T >
void write2File(T& data, const std::string& name, T_Accessor acc = T_Accessor()){
    auto copy = foobar::policies::makeCopy(acc, foobar::policies::StringStreamAccessor<>());

    foobar::types::AddDimsWrapper< std::ofstream, 2 > inFile(name.c_str());
    copy(data, inFile);
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

struct CalcIntensityFunc
{
    template< typename T, typename = std::enable_if_t< foobar::traits::IsComplex<T>::value > >
    auto
    operator()(const T& val) const
    -> decltype(val.real*val.real + val.imag*val.imag)
    {
        return val.real*val.real + val.imag*val.imag;
    }

    template< typename T, typename = std::enable_if_t< foobar::traits::IsComplex<T>::value > >
    auto
    operator()(const T& val) const
    -> decltype(val[0]*val[0] + val[1]*val[1])
    {
        return val[0]*val[0] + val[1]*val[1];
    }

    template< typename T, typename = std::enable_if_t< !foobar::traits::IsComplex<T>::value > >
    auto
    operator()(const T& val) const
    -> decltype(val*val)
    {
        return val*val;
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
    using FFT_Type = foobar::FFT_2D_C2C_D;
    auto input = foobar::wrapFFT_Input(FFT_Type(), aperture, foobar::policies::VolumeAccessor());
    auto output = foobar::wrapFFT_Output(FFT_Type(), fftResult, foobar::policies::VolumeAccessor());
    auto fft = foobar::makeFFT<foobar::libraries::fftw::FFTW<>>(input, output);
	generateData(aperture, Rect<double>(20,20));
	fft(input, output);
	using IntensityAcc =
	        foobar::policies::TransformAccessor<
                foobar::policies::VolumeAccessor,
                CalcIntensityFunc
            >;
	write2File(aperture, "input.txt", IntensityAcc());
	write2File(fftResult, "output.txt", foobar::policies::TransposeAccessor<IntensityAcc>());
}

void testReal()
{
    RealVol2D aperture(1024, 1024);
    ComplexVolFFTW2D fftResult(aperture.xDim()/2+1, aperture.yDim(), aperture.zDim());
    RealVol2D intensity(aperture.xDim(), aperture.yDim(), aperture.zDim());
    using FFT_Type = foobar::FFT_2D_R2C_D;
    auto input = foobar::wrapFFT_Input(FFT_Type(), aperture, foobar::policies::VolumeAccessor());
    auto output = foobar::wrapFFT_Output(FFT_Type(), fftResult, foobar::policies::VolumeAccessor());
    auto fft = foobar::makeFFT<foobar::libraries::fftw::FFTW<>>(input, output);
    generateData(aperture, Rect<double>(20,20));
    write2File<foobar::policies::VolumeAccessor>(aperture, "input.txt");
    fft(input, output);
    DimOffsetWrapper< SymetricAdapter<fftw_complex>, 1 > symAdapter(aperture.xDim(), fftResult);
    using CopyIntensity = foobar::policies::Copy<
        foobar::policies::TransformAccessor<
            foobar::policies::VolumeAccessor,
            CalcIntensityFunc
        >,
        foobar::policies::VolumeAccessor
    >;
    CopyIntensity copy;
    copy(symAdapter, intensity);
    auto adapter = makeTransposeAdapter(intensity);
    write2File<foobar::policies::VolumeAccessor>(adapter, "output.txt");
}

template< typename T_File >
void testFile( T_File& file )
{
    using FFTResult_t = DimOffsetWrapper< Volume< MyComplex<float> >, 1 >;
    FFTResult_t fftResult(file.getExtents()[0]/2+1, file.getExtents()[1]);
    using FFT_Type = foobar::FFT_2D_R2C_F;
    auto input = foobar::wrapFFT_Input(FFT_Type(), file);
    auto output = foobar::wrapFFT_Output(FFT_Type(), fftResult, foobar::policies::VolumeAccessor());
#ifdef WITH_CUDA
    using Library = foobar::libraries::cuFFT::CuFFT<>;
#else
    using Library = foobar::libraries::fftw::FFTW<>;
#endif
    auto fft = foobar::makeFFT<Library>(input, output);
    file.loadData(true);
    fft(input, output);
    foobar::types::SymmetricWrapper< decltype(fftResult), foobar::policies::VolumeAccessor > fullResult(fftResult, file.getExtents()[0]);
    using IntensityAcc =
        foobar::policies::TransformAccessor<
            foobar::policies::TransposeAccessor<>,
            CalcIntensityFunc
        >;
    write2File<foobar::policies::DataContainerAccessor>(file.getData(), "input.txt");
    write2File<IntensityAcc>(fullResult, "output.txt");
}

/*
 *
 */
int main(int argc, char** argv) {
    //test();
    //testIntensityCalculator();
    //testReal();
    //testComplex();
    using FileType = foobar::types::FileContainer<
        libTiff::TiffImage<>,
        foobar::policies::ImageAccessorGetColorAsFp<>,
        float,
        false
        >;
    FileType myFile("rect.tif");
    testFile(myFile);
    if(std::system("python writeData.py -i input.txt -o input.pdf"))
        std::cout << "Error converting input\n";
    if(std::system("python writeData.py -s -i output.txt -o output.pdf"))
        std::cout << "Error converting output\n";
    return 0;
}

