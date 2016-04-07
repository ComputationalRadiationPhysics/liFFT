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
 
#include "testZip.hpp"
#include "testUtils.hpp"
#include "libLiFFT/accessors/ZipAccessor.hpp"
#include "libLiFFT/generateData.hpp"
#include "libLiFFT/FFT.hpp"

#include "tiffWriter/image.hpp"
#include "tiffWriter/traitsAndPolicies.hpp"

#include <algorithm>
#include <iostream>

using LiFFT::generateData;
using namespace LiFFT::generators;

namespace LiFFTTest {

    void testZipBasic()
    {
        TestExtents extents = TestExtents::all(testSize);
        BaseR2CInput input1(extents);
        BaseR2CInput input2(extents);
        BaseR2CInput inputProd(extents);
        generateData(input1, Rect<TestPrecision>(20, testSize/2));
        generateData(input2, Cosinus<TestPrecision>(testSize, testSize/2));
        generateData(inputProd, SetToConst<TestPrecision>(0));
        TestExtents idx = TestExtents::all(0);
        for(idx[0] = 0; idx[0]<testSize; idx[0]++)
            for(idx[1] = 0; idx[1]<testSize; idx[1]++)
                inputProd(idx) = input1(idx) * input2(idx);
        auto acc = LiFFT::accessors::makeZipAccessor(input1, std::multiplies<LiFFT::types::Real<TestPrecision>>(), LiFFT::traits::getIdentityAccessor(input2));
        using FFT_Type = LiFFT::FFT_2D_R2C<TestPrecision>;
        auto input = FFT_Type::wrapInput(input2, acc);
        auto output = FFT_Type::createNewOutput(input);
        auto fft = LiFFT::makeFFT<TestLibrary>(input, output);
        fft(input, output);
        LiFFT::policies::copy(inputProd, baseR2CInput);
        execBaseR2C();
        checkResult(baseR2COutput, output, "R2C with zip accessor");
    }

    void testZipFile(const std::string& filePath1, const std::string& filePath2)
    {
        tiffWriter::FloatImage<> img1(filePath1, false);
        tiffWriter::FloatImage<> img2(filePath2, false);
        auto acc = LiFFT::accessors::makeZipAccessor(img1, std::multiplies<LiFFT::types::Real<TestPrecision>>(), LiFFT::traits::getIdentityAccessor(img2));
        using FFT_Type = LiFFT::FFT_2D_R2C_F<>;
        auto input = FFT_Type::wrapInput(img2, acc);
        auto output = FFT_Type::createNewOutput(input);
        auto fft = LiFFT::makeFFT< TestLibrary, false >(input, output);
        img1.load(); img2.load();
        LiFFT::policies::copy(img2, baseR2CInput, acc);
        fft(input, output);
        execBaseR2C();
        checkResult(baseR2COutput, output, "TIFF-ZIP test");
        visualizeOutput(BaseInstance::OutR2C, "Tiff-Zip.pdf");
    }

    void testZip(){
        testZipBasic();
        testZipFile("input1.tif", "input2.tif");
    }

}  // namespace LiFFTTest
