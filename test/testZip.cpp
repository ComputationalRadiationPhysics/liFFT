/* This file is part of HaLT.
 *
 * HaLT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * HaLT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with HaLT.  If not, see <www.gnu.org/licenses/>.
 */
 
#include "testZip.hpp"
#include "testUtils.hpp"
#include "haLT/accessors/ZipAccessor.hpp"
#include "haLT/generateData.hpp"
#include "haLT/FFT.hpp"

#include "tiffWriter/image.hpp"
#include "tiffWriter/traitsAndPolicies.hpp"

#include <algorithm>
#include <iostream>

using haLT::generateData;
using namespace haLT::generators;

namespace haLTTest {

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
        auto acc = haLT::accessors::makeZipAccessor(input1, std::multiplies<haLT::types::Real<TestPrecision>>(), haLT::traits::getIdentityAccessor(input2));
        using FFT_Type = haLT::FFT_2D_R2C<TestPrecision>;
        auto input = FFT_Type::wrapInput(input2, acc);
        auto output = FFT_Type::createNewOutput(input);
        auto fft = haLT::makeFFT<TestLibrary>(input, output);
        fft(input, output);
        haLT::policies::copy(inputProd, baseR2CInput);
        execBaseR2C();
        checkResult(baseR2COutput, output, "R2C with zip accessor");
    }

    void testZipFile(const std::string& filePath1, const std::string& filePath2)
    {
        tiffWriter::FloatImage<> img1(filePath1, false);
        tiffWriter::FloatImage<> img2(filePath2, false);
        auto acc = haLT::accessors::makeZipAccessor(img1, std::multiplies<haLT::types::Real<TestPrecision>>(), haLT::traits::getIdentityAccessor(img2));
        using FFT_Type = haLT::FFT_2D_R2C_F<>;
        auto input = FFT_Type::wrapInput(img2, acc);
        auto output = FFT_Type::createNewOutput(input);
        auto fft = haLT::makeFFT< TestLibrary, false >(input, output);
        img1.load(); img2.load();
        haLT::policies::copy(img2, baseR2CInput, acc);
        fft(input, output);
        execBaseR2C();
        checkResult(baseR2COutput, output, "TIFF-ZIP test");
        visualizeOutput(BaseInstance::OutR2C, "Tiff-Zip.pdf");
    }

    void testZip(){
        testZipBasic();
        testZipFile("input1.tif", "input2.tif");
    }

}  // namespace haLTTest
