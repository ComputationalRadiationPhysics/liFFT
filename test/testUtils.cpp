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

namespace foobarTest {

    BaseR2CInput  baseR2CInput;
    BaseR2COutput baseR2COutput;
    BaseC2CInput  baseC2CInput;
    BaseC2COutput baseC2COutput;

    using FFT_R2C = foobar::FFT_Definition< foobar::FFT_Kind::Real2Complex, testNumDims, TestPrecision, std::true_type>;
    using FFT_C2C = foobar::FFT_Definition< foobar::FFT_Kind::Complex2Complex, testNumDims, TestPrecision, std::true_type>;

    foobar::FFT_Interface_Outplace<decltype(FFT_R2C::wrapFFT_Input(baseR2CInput)), decltype(FFT_R2C::wrapFFT_Output(baseR2COutput))>* fftR2C;
    foobar::FFT_Interface_Outplace<decltype(FFT_C2C::wrapFFT_Input(baseC2CInput)), decltype(FFT_C2C::wrapFFT_Output(baseC2COutput))>* fftC2C;

    /**
     * Writes nD data to a file as strings

     * @param name filePath
     * @param data Data to write
     * @param acc  Accessor to use
     */
    template< typename T, class T_Accessor = foobar::traits::DefaultAccessor_t<T> >
    void write2File(const std::string& name, T& data, T_Accessor acc = T_Accessor()){
        auto copy = foobar::policies::makeCopy(acc, foobar::policies::StringStreamAccessor<>());

        foobar::types::AddDimsWrapper< std::ofstream, 2 > file(name.c_str());
        copy(data, file);
        file.close();
    }

    /**
     * Writes the intensity of the data to a file (as string)
     *
     * @param name filePath
     * @param data Data to write
     * @param acc  Accessor to use
     */
    template< typename T, class T_Accessor = foobar::traits::DefaultAccessor_t<T> >
    void writeIntensity2File(const std::string& name, T& data, T_Accessor acc = T_Accessor()){
        write2File(name, data, foobar::policies::makeTransformAccessor(acc, foobar::policies::CalcIntensityFunc()));
    }

    void init()
    {
        TestExtents size = TestExtents::all(testSize);
        TestExtents sizeHalf = size;
        sizeHalf[testNumDims-1] = size[testNumDims-1] / 2 + 1;
        baseR2CInput.extents = size;
        baseR2COutput.extents = sizeHalf;
        baseC2CInput.extents = size;
        baseC2COutput.extents = size;
        baseR2CInput.allocData();
        baseR2COutput.allocData();
        baseC2CInput.allocData();
        baseC2COutput.allocData();

        {
            auto input = FFT_R2C::wrapFFT_Input(baseR2CInput);
            auto output = FFT_R2C::wrapFFT_Output(baseR2COutput);
            using FFT = decltype(foobar::makeFFT<TestLibrary>(input, output));
            fftR2C = static_cast<decltype(fftR2C)>(malloc(sizeof(FFT)));
            new(fftR2C)auto(FFT(input, output));
        }
        {
            auto input = FFT_C2C::wrapFFT_Input(baseC2CInput);
            auto output = FFT_C2C::wrapFFT_Output(baseC2COutput);
            using FFT = decltype(foobar::makeFFT<TestLibrary>(input, output));
            fftC2C = static_cast<decltype(fftC2C)>(malloc(sizeof(FFT)));
            new(fftC2C)auto(FFT(input, output));
        }

    }

    void finalize()
    {
        baseR2CInput.freeData();
        baseR2COutput.freeData();
        baseC2CInput.freeData();
        baseC2COutput.freeData();
        delete fftR2C;
        delete fftC2C;
    }

    void visualizeBase()
    {
        generateData(baseR2CInput, Rect<TestPrecision>(20,20,500,500));
        generateData(baseC2CInput, Rect<TestPrecision>(20,20,500,500));
        execBaseR2C();
        execBaseC2C();
        writeIntensity2File("inputR2C.txt", baseR2CInput);
        writeIntensity2File("inputC2C.txt", baseC2CInput);
        auto fullR2COutput = foobar::types::makeSymmetricWrapper(baseR2COutput, baseC2CInput.extents[baseR2CInput.numDims-1]);
        writeIntensity2File("outputR2C.txt", fullR2COutput, foobar::policies::makeTransposeAccessorFor(fullR2COutput));
        writeIntensity2File("outputC2C.txt", baseC2COutput, foobar::policies::makeTransposeAccessorFor(baseC2COutput));
        auto e = compare(fullR2COutput, baseC2COutput);
        if(!e.first)
            std::cerr << "Test output mismatch: " << e.second << std::endl;
        else
            std::cout << "Self-check passed" << std::endl;
        if(std::system("python writeData.py -i inputR2C.txt -o inputR2C.pdf"))
            std::cout << "Error converting input R2C\n";
        if(std::system("python writeData.py -i inputC2C.txt -o inputC2C.pdf"))
            std::cout << "Error converting input C2C\n";
        if(std::system("python writeData.py -s -i outputR2C.txt -o outputR2C.pdf"))
            std::cout << "Error converting output R2C\n";
        if(std::system("python writeData.py -s -i outputC2C.txt -o outputC2C.pdf"))
            std::cout << "Error converting output C2C\n";
    }

    void execBaseR2C()
    {
        auto input  = FFT_R2C::wrapFFT_Input( baseR2CInput);
        auto output = FFT_R2C::wrapFFT_Output(baseR2COutput);
        (*fftR2C)(input, output);
    }

    void execBaseC2C()
    {
        auto input  = FFT_C2C::wrapFFT_Input( baseC2CInput);
        auto output = FFT_C2C::wrapFFT_Output(baseC2COutput);
        (*fftC2C)(input, output);
    }

}  // namespace foobarTest
