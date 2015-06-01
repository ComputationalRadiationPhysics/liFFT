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
#include "policyTest.hpp"
#include "IntensityCalculator_Test.hpp"

template<typename T=double>
struct MyComplex{
	T real, imag;
	MyComplex(){}
	MyComplex(T real): real(real), imag(0){}
};

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

template<typename T=double>
class Volume{
    T* data_;
    bool isOwned_;
    const size_t xDim_, yDim_, zDim_;
public:
    using value_type = T;

    Volume(size_t xDim, size_t yDim, size_t zDim): xDim_(xDim), yDim_(yDim), zDim_(zDim){
        data_ = static_cast<T*>(fftw_malloc(xDim*yDim*zDim*sizeof(T))); //new T[xDim*yDim*zDim_];
        isOwned_ = true;
    }
    Volume(size_t xDim, size_t yDim, size_t zDim, T* data): xDim_(xDim), yDim_(yDim), zDim_(zDim){
        data_ = data;
        isOwned_ = false;
    }
    ~Volume(){
        if(isOwned_)
            fftw_free(data_);//delete[] data_;
    }
    T* data(){
        return data_;
    }
    T& operator()(size_t x, size_t y=0, size_t z=0){
        return data_[(z*yDim_ + y)*xDim_ + x];
    }
    const T& operator()(size_t x, size_t y=0, size_t z=0) const{
        return data_[(z*yDim_ + y)*xDim_ + x];
    }

    size_t xDim() const{ return xDim_; }
    size_t yDim() const{ return yDim_; }
    size_t zDim() const{ return zDim_; }
};

template<typename T>
class VolumeAdapter{
protected:
    T& obj_;
public:
    using value_type = typename T::value_type;
    VolumeAdapter(T& obj):obj_(obj){}

    size_t xDim() const{ return obj_.xDim(); }
    size_t yDim() const{ return obj_.yDim(); }
    size_t zDim() const{ return obj_.zDim(); }
};

template<typename T>
class SymetricAdapter: public Volume<T>{
    const size_t realXDim;
public:
    using parent_type = Volume<T>;
    SymetricAdapter(size_t xDim, size_t yDim, size_t zDim): parent_type(xDim/2+1, yDim, zDim), realXDim(xDim){}
    SymetricAdapter(size_t xDim, size_t yDim, size_t zDim, T* data): parent_type(xDim/2+1, yDim, zDim, data), realXDim(xDim){}
    SymetricAdapter(size_t realXDim, parent_type& data): parent_type(data.xDim(), data.yDim(), data.zDim(), data.data()), realXDim(realXDim){}

    T& operator()(size_t x, size_t y=0, size_t z=0){
        x = (x>=parent_type::xDim()) ? realXDim-x : x;
        return parent_type::operator ()(x, y, z);
    }
    const T& operator()(size_t x, size_t y=0, size_t z=0) const{
        x = (x>=parent_type::xDim()) ? realXDim-x : x;
        return parent_type::operator ()(x, y, z);
    }
    size_t xDim() const{ return realXDim; }
};

template<typename T>
class TransposeAdapter: public VolumeAdapter<T>{
public:
    using value_type = typename T::value_type;
    TransposeAdapter(T& obj):VolumeAdapter<T>(obj){}
    value_type& operator()(size_t x, size_t y=0, size_t z=0){
        x = (x>=this->xDim()/2) ? x-this->xDim()/2 : x+this->xDim()/2;
        y = (y>=this->yDim()/2) ? y-this->yDim()/2 : y+this->yDim()/2;
        z = (z>=this->zDim()/2) ? z-this->zDim()/2 : z+this->zDim()/2;
        return this->obj_(x, y, z);
    }
    const value_type& operator()(size_t x, size_t y=0, size_t z=0) const{
        x = (x>=this->xDim()/2) ? x-this->xDim()/2 : x+this->xDim()/2;
        y = (y>=this->yDim()/2) ? y-this->yDim()/2 : y+this->yDim()/2;
        z = (z>=this->zDim()/2) ? z-this->zDim()/2 : z+this->zDim()/2;
        return this->obj_(x, y, z);
    }
};

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

template<typename T> inline
TransposeAdapter<T> makeTransposeAdapter(T& obj){
    return TransposeAdapter<T>(obj);
}

template<class Generator, typename T>
void generateData(Volume<T>& data){
    for(size_t z=0; z<data.zDim(); ++z){
        for(size_t y=0; y<data.yDim(); ++y){
            for(size_t x=0; x<data.xDim(); ++x)
                data(x,y,z) = Generator::template apply<T>(x,y,z);
        }
    }
}

template<int size>
struct Spalt{
    template<typename T>
    static T apply(size_t x, size_t y, size_t z){
        return (abs(x-500)<size) ? 1 : 0;
    }
};

template<int sizeX, int sizeY>
struct Rect{
    template<typename T>
    static T apply(size_t x, size_t y, size_t z){
        return (abs(x-500)<sizeX && abs(y-500)<sizeY) ? 1 : 0;
    }
};

template<int size>
struct Circle{
    template<typename T>
    static T apply(size_t x, size_t y, size_t z){
        return (pow(abs(x-500), 2)+pow(abs(y-500), 2)<size*size) ? 1 : 0;
    }
};

struct Nullify{
    template<typename T>
    static T apply(size_t x, size_t y, size_t z){
        return 0;
    }
};

template<typename T, typename T2>
void calcIntensities(const T& in, T2& out, size_t numX, size_t numY){
    for(size_t y=0; y<numY; ++y){
        for(size_t x=0; x<numX; ++x){
            out(x,y) = absSqr(in(x,y));
        }
    }
}

void testComplex(){
	Volume<MyComplex<> > aperture(1024, 1024, 1);
	Volume<MyComplex<> > fftResult(aperture.xDim(), aperture.yDim(), aperture.zDim());
	Volume<double> intensity(aperture.xDim(), aperture.yDim(), aperture.zDim());
    fftw_plan plan = fftw_plan_dft_2d(aperture.yDim(), aperture.xDim(), reinterpret_cast<fftw_complex*>(aperture.data()), reinterpret_cast<fftw_complex*>(fftResult.data()), FFTW_FORWARD, FFTW_ESTIMATE);
	generateData<Spalt<5> >(aperture);
	fftw_execute(plan);
	fftw_destroy_plan(plan);
	calcIntensities(aperture, intensity, aperture.xDim(), aperture.yDim());
	write2File(intensity, "input.txt");
	calcIntensities(fftResult, intensity, fftResult.xDim(), fftResult.yDim());
	auto adapter = makeTransposeAdapter(intensity);
	write2File(adapter, "output.txt");
}

void testReal(){
    Volume<> aperture(1024, 1024, 1);
    Volume<fftw_complex> fftResult(aperture.xDim()/2+1, aperture.yDim(), aperture.zDim());
    Volume<> intensity(aperture.xDim(), aperture.yDim(), aperture.zDim());
    fftw_plan plan = fftw_plan_dft_r2c_2d(aperture.yDim(), aperture.xDim(), aperture.data(), reinterpret_cast<fftw_complex*>(fftResult.data()), FFTW_ESTIMATE);
    generateData<Rect<20,20> >(aperture);
    write2File(aperture, "input.txt");
    fftw_execute(plan);
    fftw_destroy_plan(plan);
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
    return 0;
    //testComplex();
    testReal();
    if(std::system("python writeData.py -i input.txt -o input.pdf"));
    if(std::system("python writeData.py -s -i output.txt -o output.pdf"));
    return 0;
}

