#include "testUtils.hpp"
#include "foobar/types/Vec.hpp"
#include "foobar/FFT.hpp"
#include "foobar/libraries/fftw/FFTW.hpp"
#include "generateData.hpp"
#include "foobar/accessors/TransformAccessor.hpp"
#include "foobar/accessors/TransposeAccessor.hpp"
#include "foobar/types/SymmetricWrapper.hpp"
#include "foobar/accessors/StreamAccessor.hpp"
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
        auto copy = foobar::policies::makeCopy(acc, foobar::accessors::StringStreamAccessor<>());

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
        write2File(name, data, foobar::accessors::makeTransformAccessor(acc, foobar::policies::CalcIntensityFunc()));
    }

    void init()
    {
        TestExtents size = TestExtents::all(testSize);
        TestExtents sizeHalf = size;
        sizeHalf[testNumDims-1] = size[testNumDims-1] / 2 + 1;
        baseR2CInput.allocData(size);
        baseR2COutput.allocData(sizeHalf);
        baseC2CInput.allocData(size);
        baseC2COutput.allocData(size);

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
        generateData(baseR2CInput, Rect<TestPrecision>(20,testSize/2));
        generateData(baseC2CInput, Rect<TestPrecision>(20,testSize/2));
        execBaseR2C();
        execBaseC2C();
        auto fullR2COutput = foobar::types::makeSymmetricWrapper(baseR2COutput, baseC2CInput.getExtents()[baseR2CInput.numDims-1]);
        auto e = compare(fullR2COutput, baseC2COutput);
        if(!e.first)
            std::cerr << "Test output mismatch: " << e.second << std::endl;
        else
            std::cout << "Self-check passed" << std::endl;
        visualizeOutput(BaseInstance::InC2C, "inputC2C.pdf");
        visualizeOutput(BaseInstance::InR2C, "inputR2C.pdf");
        visualizeOutput(BaseInstance::OutC2C, "outputC2C.pdf");
        visualizeOutput(BaseInstance::OutR2C, "outputR2C.pdf");
    }

    void visualizeOutput(BaseInstance inst, const std::string& filePath){
        std::string txtFile = filePath+".txt";
        switch(inst)
        {
        case BaseInstance::InC2C:
            writeIntensity2File(txtFile, baseC2CInput);
            break;
        case BaseInstance::InR2C:
            writeIntensity2File(txtFile, baseC2CInput);
            break;
        case BaseInstance::OutC2C:
            writeIntensity2File(txtFile, baseC2COutput, foobar::accessors::makeTransposeAccessorFor(baseC2COutput));
            break;
        case BaseInstance::OutR2C:
            {
                auto fullR2COutput = foobar::types::makeSymmetricWrapper(baseR2COutput, baseC2CInput.getExtents()[baseR2CInput.numDims-1]);
                writeIntensity2File(txtFile, fullR2COutput, foobar::accessors::makeTransposeAccessorFor(fullR2COutput));
            }
            break;
        default:
            throw std::runtime_error("Invalid value");
        }
        std::string cmd = "python writeData.py -i " + txtFile + " -o " + filePath;
        if(std::system(cmd.c_str()))
            std::cout << "Error converting txt to pdf for " << filePath << "\n";
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
