/* 
 * File:     main.cpp
 * Author: grund59
 *
 * Created on 13. Mai 2015, 13:23
 */

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>
#include <fftw3.h>
#include "c++14_types.hpp"
#include "policyTest.hpp"
#include "Volume.hpp"
#include "VolumeAdapter.hpp"
#include "generateData.hpp"
#include "IntensityCalculator_Test.hpp"
#include "foobar/libraries/fftw/FFTW.hpp"
#include "foobar/policies/FFT.hpp"

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

template<typename T>
void write2File(const T& data, const std::string& name){
    std::ofstream inFile(name.c_str());
    for(size_t y=0; y<data.yDim(); ++y){
        for(size_t x=0; x<data.xDim(); ++x){
            inFile << data(x,y);
            if(x+1<data.xDim())
                inFile << " ";
        }
        if(y+1<data.xDim())
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
using RealVol = Volume<double>;

void testComplex(){
    ComplexVol aperture(1024, 1024, 1);
    ComplexVol fftResult(aperture.xDim(), aperture.yDim(), aperture.zDim());
	RealVol intensity(aperture.xDim(), aperture.yDim(), aperture.zDim());
    //fftw_plan plan = fftw_plan_dft_2d(aperture.yDim(), aperture.xDim(), reinterpret_cast<fftw_complex*>(aperture.data()), reinterpret_cast<fftw_complex*>(fftResult.data()), FFTW_FORWARD, FFTW_ESTIMATE);
    using FFTType = typename foobar::policies::FFT< foobar::libraries::fftw::FFTW, true, ComplexVol, ComplexVol, std::integral_constant<unsigned, 2> >::type;
    FFTType fft(aperture, fftResult);
	generateData(aperture, Spalt<double>(5));
	//fftw_execute(plan);
	//fftw_destroy_plan(plan);
	fft();
	calcIntensities(aperture, intensity, aperture.xDim(), aperture.yDim());
	write2File(intensity, "input.txt");
	calcIntensities(fftResult, intensity, fftResult.xDim(), fftResult.yDim());
	auto adapter = makeTransposeAdapter(intensity);
	write2File(adapter, "output.txt");
}

void testReal(){
    RealVol aperture(1024, 1024, 1);
    ComplexVolFFTW fftResult(aperture.xDim()/2+1, aperture.yDim(), aperture.zDim());
    RealVol intensity(aperture.xDim(), aperture.yDim(), aperture.zDim());
    //fftw_plan plan = fftw_plan_dft_r2c_2d(aperture.yDim(), aperture.xDim(), aperture.data(), reinterpret_cast<fftw_complex*>(fftResult.data()), FFTW_ESTIMATE);
    using FFTType = typename foobar::policies::FFT< foobar::libraries::fftw::FFTW, true, RealVol, ComplexVolFFTW, std::integral_constant<unsigned, 2> >::type;
    FFTType fft(aperture, fftResult);
    generateData(aperture, Rect<double>(20,20));
    write2File(aperture, "input.txt");
    //fftw_execute(plan);
    //fftw_destroy_plan(plan);
    fft();
    SymetricAdapter<fftw_complex> symAdapter(aperture.xDim(), fftResult);
    calcIntensities(symAdapter, intensity, symAdapter.xDim(), symAdapter.yDim());
    auto adapter = makeTransposeAdapter(intensity);
    write2File(adapter, "output.txt");
}

/*
 *
 */
int main(int argc, char** argv) {
    //test();
    testIntensityCalculator();
    //testComplex();
    testReal();
    if(std::system("python writeData.py -i input.txt -o input.pdf"));
    if(std::system("python writeData.py -s -i output.txt -o output.pdf"));
    return 0;
}

