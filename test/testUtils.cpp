/* This file is part of libLiFFT.
 *
 * libLiFFT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * libLiFFT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with libLiFFT.  If not, see <www.gnu.org/licenses/>.
 */
 
#include "testUtils.hpp"
#include "libLiFFT/types/Vec.hpp"
#include "libLiFFT/FFT.hpp"
#include "libLiFFT/generateData.hpp"
#include "libLiFFT/accessors/TransformAccessor.hpp"
#include "libLiFFT/accessors/TransposeAccessor.hpp"
#include "libLiFFT/types/SymmetricWrapper.hpp"
#include "libLiFFT/accessors/StreamAccessor.hpp"
#include "libLiFFT/types/AddDimsWrapper.hpp"
#include "libLiFFT/policies/CalcIntensityFunctor.hpp"
#include "libLiFFT/types/View.hpp"
#include "libLiFFT/types/SliceView.hpp"
#include <iostream>
#include <fstream>

using LiFFT::generateData;
using namespace LiFFT::generators;

namespace LiFFTTest {

    BaseR2CInput  baseR2CInput;
    BaseR2COutput baseR2COutput;
    BaseC2CInput  baseC2CInput;
    BaseC2COutput baseC2COutput;

    using FFT_R2C = LiFFT::FFT_Definition< LiFFT::FFT_Kind::Real2Complex, testNumDims, TestPrecision, std::true_type>;
    using FFT_C2C = LiFFT::FFT_Definition< LiFFT::FFT_Kind::Complex2Complex, testNumDims, TestPrecision, std::true_type>;

    LiFFT::FFT_Interface_Outplace<decltype(FFT_R2C::wrapInput(baseR2CInput)), decltype(FFT_R2C::wrapOutput(baseR2COutput))>* fftR2C;
    LiFFT::FFT_Interface_Outplace<decltype(FFT_C2C::wrapInput(baseC2CInput)), decltype(FFT_C2C::wrapOutput(baseC2COutput))>* fftC2C;

    /**
     * Writes nD data to a file as strings

     * @param name filePath
     * @param data Data to write
     * @param acc  Accessor to use
     */
    template< typename T, class T_Accessor = LiFFT::traits::IdentityAccessor_t<T> >
    void write2File(const std::string& name, T& data, T_Accessor acc = T_Accessor()){
        auto copy = LiFFT::policies::makeCopy(acc, LiFFT::accessors::StringStreamAccessor<>());

        LiFFT::types::AddDimsWrapper< std::ofstream, 2 > file(name.c_str());
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
    template< typename T, class T_Accessor = LiFFT::traits::IdentityAccessor_t<T> >
    void writeIntensity2File(const std::string& name, T& data, T_Accessor acc = T_Accessor()){
        write2File(name, data, LiFFT::accessors::makeTransformAccessor(acc, LiFFT::policies::CalcIntensityFunc()));
    }

    void
    testView()
    {
        using Extents = LiFFT::types::Vec<2>;
        const Extents size(100u, 120u);
        const Extents offset(10u, 5u);
        const Extents viewSize(40u, 65u);
        auto data = LiFFT::mem::RealContainer<2, float>(size);
        auto view = LiFFT::types::makeView(data, LiFFT::types::makeRange(offset, viewSize));
        Extents idx, viewIdx;
        bool error = false;
        for(idx[0] = offset[0], viewIdx[0] = 0; viewIdx[0] < viewSize[0]; ++idx[0], ++viewIdx[0])
            for(idx[1] = offset[1], viewIdx[1] = 0; viewIdx[1] < viewSize[1]; ++idx[1], ++viewIdx[1])
            {
                float value = rand() / RAND_MAX;
                view(viewIdx) = value;
                if(data(idx) != view(viewIdx) || data(idx) != value)
                    error = true;
            }
        if(error)
            std::cerr << "View test failed" << std::endl;
        else
            std::cout << "View test passed" << std::endl;
    }

    unsigned getIdx(unsigned idx, unsigned fixedDim)
    {
        return idx >= fixedDim ? idx + 1 : idx;
    }

    template< unsigned fixedDim >
    void
    testSliceView()
    {
        using Extents3 = LiFFT::types::Vec<3>;
        using Extents2 = LiFFT::types::Vec<2>;
        const Extents3 size(100u, 120u, 214u);
        const Extents3 offset(10u, 5u, 23u);
        const Extents2 viewSize(40u, 65u);
        auto data = LiFFT::mem::RealContainer<3, float>(size);
        auto view = LiFFT::types::makeSliceView<fixedDim>(data, LiFFT::types::makeRange(offset, viewSize));
        Extents3 idx;
        Extents2 viewIdx;
        bool error = false;
        idx[fixedDim] = offset[fixedDim];
        unsigned idxOut = getIdx(0, fixedDim);
        unsigned idxIn = getIdx(1, fixedDim);
        for(idx[idxOut] = offset[idxOut], viewIdx[0] = 0; viewIdx[0] < viewSize[0]; ++idx[idxOut], ++viewIdx[0])
            for(idx[idxIn] = offset[idxIn], viewIdx[1] = 0; viewIdx[1] < viewSize[1]; ++idx[idxIn], ++viewIdx[1])
            {
                float value = rand() / RAND_MAX;
                view(viewIdx) = value;
                if(data(idx) != view(viewIdx) || data(idx) != value)
                    error = true;
            }
        if(error)
            std::cerr << "SliceView(" << fixedDim << ") test failed" << std::endl;
        else
            std::cout << "SliceView(" << fixedDim << ") test passed" << std::endl;
    }

    void
    testDataWrappers()
    {
        const unsigned size = 100u;
        using Extents = LiFFT::types::Vec<3>;
        using FFT = LiFFT::FFT_3D_R2C_F<>;
        auto input = FFT::wrapInput(
                        LiFFT::mem::RealContainer<3, float>(
                                Extents(size, size, size)
                        )
                     );
        auto data = LiFFT::mem::RealContainer<3, float>(
                Extents(size, size, size)
        );
        auto output = FFT::wrapInput(data);

        Extents idx = Extents::all(0u);
        const float val = 1337;
        const float val2 = 1338;
        bool error = false;
        auto acc = LiFFT::traits::getIdentityAccessor(input);
        auto acc2 = LiFFT::traits::getIdentityAccessor(output);
        auto acc3 = LiFFT::traits::getIdentityAccessor(data);
        for(unsigned i=0; i<4; i++){
            input(idx) = val;
            output(idx) = val;
            if(input(idx) != val)
                error = true;
            if(output(idx) != val || acc3(idx, data) != val)
                error = true;
            acc(idx, input) = val2;
            acc2(idx, output) = val2;
            if(acc(idx, input) != val2 || input(idx)!=val2)
                error = true;
            if(acc2(idx, output) != val2 || acc3(idx, data) != val2 || output(idx)!=val2)
                error = true;
            if(i<3)
                idx[i] = size/2;
        }
        if(error)
            std::cerr << "Datawrapper test failed" << std::endl;
        else
            std::cout << "Datawrapper test passed" << std::endl;
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
            auto input = FFT_R2C::wrapInput(baseR2CInput);
            auto output = FFT_R2C::wrapOutput(baseR2COutput);
            using FFT = decltype(LiFFT::makeFFT<TestLibrary, false>(input, output));
            fftR2C = static_cast<decltype(fftR2C)>(malloc(sizeof(FFT)));
            new(fftR2C)auto(FFT(input, output));
        }
        {
            auto input = FFT_C2C::wrapInput(baseC2CInput);
            auto output = FFT_C2C::wrapOutput(baseC2COutput);
            using FFT = decltype(LiFFT::makeFFT<TestLibrary, false>(input, output));
            fftC2C = static_cast<decltype(fftC2C)>(malloc(sizeof(FFT)));
            new(fftC2C)auto(FFT(input, output));
        }
        testDataWrappers();
        testView();
        testSliceView<0>();
        testSliceView<1>();
        testSliceView<2>();
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
        auto fullR2COutput = LiFFT::types::makeSymmetricWrapper(baseR2COutput, baseC2CInput.getExtents()[baseR2CInput.numDims-1]);
        checkResult(baseC2COutput, fullR2COutput, "Self check");

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
            writeIntensity2File(txtFile, baseC2COutput, LiFFT::accessors::makeTransposeAccessorFor(baseC2COutput));
            break;
        case BaseInstance::OutR2C:
            {
                auto fullR2COutput = LiFFT::types::makeSymmetricWrapper(baseR2COutput, baseC2CInput.getExtents()[baseR2CInput.numDims-1]);
                writeIntensity2File(txtFile, fullR2COutput, LiFFT::accessors::makeTransposeAccessorFor(fullR2COutput));
            }
            break;
        default:
            throw std::runtime_error("Invalid value");
        }
        std::string cmd = "python writeData.py -i \"" + txtFile + "\" -o \"" + filePath + "\"";
        if(std::system(cmd.c_str()))
            std::cout << "Error converting txt to pdf for " << filePath << "\n";
    }

    void execBaseR2C()
    {
        auto input  = FFT_R2C::wrapInput( baseR2CInput);
        auto output = FFT_R2C::wrapOutput(baseR2COutput);
        (*fftR2C)(input, output);
    }

    void execBaseC2C()
    {
        auto input  = FFT_C2C::wrapInput( baseC2CInput);
        auto output = FFT_C2C::wrapOutput(baseC2COutput);
        (*fftC2C)(input, output);
    }

}  // namespace LiFFTTest
