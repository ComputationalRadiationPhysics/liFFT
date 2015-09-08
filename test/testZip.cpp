#include "testZip.hpp"
#include "testUtils.hpp"
#include "foobar/accessors/ZipAccessor.hpp"
#include "foobar/generateData.hpp"
#include "foobar/FFT.hpp"

#include "tiffWriter/image.hpp"
#include "tiffWriter/traitsAndPolicies.hpp"

#include <algorithm>
#include <iostream>

using foobar::generateData;
using namespace foobar::generators;

namespace foobarTest {

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
        auto acc = foobar::accessors::makeZipAccessor(input1, std::multiplies<foobar::types::Real<TestPrecision>>(), foobar::traits::getIdentityAccessor(input2));
        using FFT_Type = foobar::FFT_2D_R2C<TestPrecision>;
        auto input = FFT_Type::wrapInput(input2, acc);
        auto output = FFT_Type::createNewOutput(input);
        auto fft = foobar::makeFFT<TestLibrary>(input, output);
        fft(input, output);
        foobar::policies::copy(inputProd, baseR2CInput);
        execBaseR2C();
        checkResult(baseR2COutput, output, "R2C with zip accessor");
    }

    void testZipFile(const std::string& filePath1, const std::string& filePath2)
    {
        tiffWriter::FloatImage<> img1(filePath1, false);
        tiffWriter::FloatImage<> img2(filePath2, false);
        auto acc = foobar::accessors::makeZipAccessor(img1, std::multiplies<foobar::types::Real<TestPrecision>>(), foobar::traits::getIdentityAccessor(img2));
        using FFT_Type = foobar::FFT_2D_R2C_F<>;
        auto input = FFT_Type::wrapInput(img2, acc);
        auto output = FFT_Type::createNewOutput(input);
        auto fft = foobar::makeFFT< TestLibrary, false >(input, output);
        img1.load(); img2.load();
        foobar::policies::copy(img2, baseR2CInput, acc);
        fft(input, output);
        execBaseR2C();
        checkResult(baseR2COutput, output, "TIFF-ZIP test");
        visualizeOutput(BaseInstance::OutR2C, "Tiff-Zip.pdf");
    }

    void testZip(){
        testZipBasic();
        testZipFile("input1.tif", "input2.tif");
    }

}  // namespace foobarTest
