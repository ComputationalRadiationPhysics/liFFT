#include "testUtils.hpp"
#include "foobar/types/Vec.hpp"
#include "foobar/FFT.hpp"
#include "foobar/libraries/fftw/FFTW.hpp"
#include "generateData.hpp"
#include "foobar/policies/TransformAccessor.hpp"
#include "foobar/policies/TransposeAccessor.hpp"
#include "foobar/types/SymmetricWrapper.hpp"
#include "foobar/policies/StreamAccessor.hpp"
#include "foobar/types/AddDimsWrapper.hpp"
#include "foobar/policies/CalcIntensityFunctor.hpp"
#include <iostream>
#include <fstream>

TestR2CInput  testR2CInput;
TestR2COutput testR2COutput;
TestC2CInput  testC2CInput;
TestC2COutput testC2COutput;

using FFT_R2C = foobar::FFT_Definition< foobar::FFT_Kind::Real2Complex, testNumDims, TestPrecision, std::true_type>;
using FFT_C2C = foobar::FFT_Definition< foobar::FFT_Kind::Complex2Complex, testNumDims, TestPrecision, std::true_type>;

foobar::FFT_Interface_Outplace<decltype(FFT_R2C::wrapFFT_Input(testR2CInput)), decltype(FFT_R2C::wrapFFT_Output(testR2COutput))>* fftR2C;
foobar::FFT_Interface_Outplace<decltype(FFT_C2C::wrapFFT_Input(testC2CInput)), decltype(FFT_C2C::wrapFFT_Output(testC2COutput))>* fftC2C;

template< typename T, class T_Accessor = foobar::traits::DefaultAccessor_t<T> >
void write2File(const std::string& name, T& data, T_Accessor acc = T_Accessor()){
    auto copy = foobar::policies::makeCopy(acc, foobar::policies::StringStreamAccessor<>());

    foobar::types::AddDimsWrapper< std::ofstream, 2 > file(name.c_str());
    copy(data, file);
    file.close();
}

template< typename T, class T_Accessor = foobar::traits::DefaultAccessor_t<T> >
void writeIntensity2File(const std::string& name, T& data, T_Accessor acc = T_Accessor()){
    write2File(name, data, foobar::policies::makeTransformAccessor(acc, foobar::policies::CalcIntensityFunc()));
}

void initTest()
{
    TestExtents size = TestExtents::all(testSize);
    TestExtents sizeHalf = size;
    sizeHalf[testNumDims-1] = size[testNumDims-1] / 2 + 1;
    testR2CInput.extents = size;
    testR2COutput.extents = sizeHalf;
    testC2CInput.extents = size;
    testC2COutput.extents = size;
    testR2CInput.allocData();
    testR2COutput.allocData();
    testC2CInput.allocData();
    testC2COutput.allocData();

    {
        auto input = FFT_R2C::wrapFFT_Input(testR2CInput);
        auto output = FFT_R2C::wrapFFT_Output(testR2COutput);
        using FFT = decltype(foobar::makeFFT<TestLibrary>(input, output));
        fftR2C = static_cast<decltype(fftR2C)>(malloc(sizeof(FFT)));
        new(fftR2C)auto(FFT(input, output));
    }
    {
        auto input = FFT_C2C::wrapFFT_Input(testC2CInput);
        auto output = FFT_C2C::wrapFFT_Output(testC2COutput);
        using FFT = decltype(foobar::makeFFT<TestLibrary>(input, output));
        fftC2C = static_cast<decltype(fftC2C)>(malloc(sizeof(FFT)));
        new(fftC2C)auto(FFT(input, output));
    }

}

void finalizeTest()
{
    testR2CInput.freeData();
    testR2COutput.freeData();
    testC2CInput.freeData();
    testC2COutput.freeData();
    delete fftR2C;
    delete fftC2C;
}

void visualizeBaseTest()
{
    generateData(testR2CInput, Rect<TestPrecision>(20,20,500,500));
    generateData(testC2CInput, Rect<TestPrecision>(20,20,500,500));
    testExecBaseR2C();
    testExecBaseC2C();
    writeIntensity2File("inputR2C.txt", testR2CInput);
    writeIntensity2File("inputC2C.txt", testC2CInput);
    auto fullR2COutput = foobar::types::makeSymmetricWrapper(testR2COutput, testC2CInput.extents[testR2CInput.numDims-1]);
    writeIntensity2File("outputR2C.txt", fullR2COutput, foobar::policies::makeTransposeAccessorFor(fullR2COutput));
    writeIntensity2File("outputC2C.txt", testC2COutput, foobar::policies::makeTransposeAccessorFor(testC2COutput));
    if(!compare(fullR2COutput, testC2COutput))
        std::cerr << "Test output mismatch";
    if(std::system("python writeData.py -i inputR2C.txt -o inputR2C.pdf"))
        std::cout << "Error converting input R2C\n";
    if(std::system("python writeData.py -i inputC2C.txt -o inputC2C.pdf"))
        std::cout << "Error converting input C2C\n";
    if(std::system("python writeData.py -s -i outputR2C.txt -o outputR2C.pdf"))
        std::cout << "Error converting output R2C\n";
    if(std::system("python writeData.py -s -i outputC2C.txt -o outputC2C.pdf"))
        std::cout << "Error converting output C2C\n";
}

void testExecBaseR2C()
{
    auto input  = FFT_R2C::wrapFFT_Input( testR2CInput);
    auto output = FFT_R2C::wrapFFT_Output(testR2COutput);
    (*fftR2C)(input, output);
}

void testExecBaseC2C()
{
    auto input  = FFT_C2C::wrapFFT_Input( testC2CInput);
    auto output = FFT_C2C::wrapFFT_Output(testC2COutput);
    (*fftC2C)(input, output);
}
