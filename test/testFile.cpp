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
 
#include "testFile.hpp"
#include "testUtils.hpp"

#include "haLT/mem/DataContainer.hpp"
#include "haLT/mem/FileContainer.hpp"
#include "haLT/accessors/ImageAccessor.hpp"
#include "tiffWriter/image.hpp"
#include "tiffWriter/traitsAndPolicies.hpp"
#include "haLT/FFT.hpp"
#include "haLT/generateData.hpp"
#include "haLT/accessors/TransposeAccessor.hpp"

using haLT::generateData;
using namespace haLT::generators;

namespace haLTTest {

    template< typename T_File >
    void testFile( T_File& file )
    {
        using FFTResult_t = haLT::mem::ComplexContainer<2, TestPrecision>;
        FFTResult_t fftResult(haLT::types::Idx2D(file.getExtents()[1], file.getExtents()[0]/2+1));
        using FFT_Type = haLT::FFT_2D_R2C<TestPrecision>;
        auto input = FFT_Type::wrapInput(file);
        auto output = FFT_Type::wrapOutput(fftResult);
        auto fft = haLT::makeFFT<TestLibrary>(input, output);
        file.loadData(true);
        fft(input, output);
        haLT::policies::copy(file, baseR2CInput);
        execBaseR2C();
        checkResult(baseR2COutput, fftResult, "R2C with file input");
    }

    void testTiffCp(const std::string& filePath)
    {
        std::string filePath2 = filePath+"2.tif";
        tiffWriter::FloatImage<> img(filePath);
        img.saveTo(filePath2);
        img.close();
        tiffWriter::FloatImage<> img1(filePath);
        tiffWriter::FloatImage<> img2(filePath2);
        checkResult(img1, img2, "TIFF copy");
        std::remove(filePath2.c_str());
    }

    void testTiffModify(const std::string& filePath)
    {
        std::string filePath2 = filePath+"2.tif";
        tiffWriter::FloatImage<> img(filePath);
        haLT::mem::RealContainer<2, float> data(haLT::types::Vec<2>(img.getHeight(), img.getWidth()));
        generateData(data, Circle<float>(50, img.getHeight()/2));
        auto acc = haLT::accessors::makeTransposeAccessorFor(img);
        haLT::policies::copy(data, img, haLT::traits::getIdentityAccessor(data), acc);
        img.saveTo(filePath2);
        img.close();
        tiffWriter::FloatImage<> img2(filePath2);
        auto accData = haLT::accessors::makeTransposeAccessorFor(data);
        auto res = compare(data, img2, CmpError(1e-8, 1e-8), accData);
        if(!res.first)
            std::cerr << "Tiff modify failed" << std::endl;
        else
            std::cout << "Tiff modify passed" << std::endl;
        std::remove(filePath2.c_str());
    }

    void testTiffFile(const std::string& filePath)
    {
        tiffWriter::FloatImage<> img(filePath, false);
        using FFT_Type = haLT::FFT_2D_R2C_F<>;
        auto input = FFT_Type::wrapInput(img);
        auto output = FFT_Type::createNewOutput(input);
        auto fft = haLT::makeFFT< TestLibrary, false >(input, output);
        img.load();
        haLT::policies::copy(img, baseR2CInput);
        fft(input, output);
        execBaseR2C();
        checkResult(baseR2COutput, output, "TIFF test");
        visualizeOutput(BaseInstance::OutR2C, "outputTiff.pdf");
    }

    void testFile()
    {
        using FileType = haLT::mem::FileContainer<
            tiffWriter::Image<>,
            haLT::accessors::ImageAccessorGetColorAsFp<TestPrecision>,
            TestPrecision,
            false
            >;
        FileType myFile("rect.tif");
        testFile(myFile);
        testTiffFile("input1.tif");
        testTiffCp("input1.tif");
        testTiffModify("input1.tif");
    }

}  // namespace haLTTest
