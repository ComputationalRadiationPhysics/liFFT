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
 
#include "testFile.hpp"
#include "testUtils.hpp"

#include "libLiFFT/mem/DataContainer.hpp"
#include "libLiFFT/mem/FileContainer.hpp"
#include "libLiFFT/accessors/ImageAccessor.hpp"
#include "tiffWriter/image.hpp"
#include "tiffWriter/traitsAndPolicies.hpp"
#include "libLiFFT/FFT.hpp"
#include "libLiFFT/generateData.hpp"
#include "libLiFFT/accessors/TransposeAccessor.hpp"

using LiFFT::generateData;
using namespace LiFFT::generators;

namespace LiFFTTest {

    template< typename T_File >
    bool testFile( T_File& file )
    {
        using FFTResult_t = LiFFT::mem::ComplexContainer<2, TestPrecision>;
        FFTResult_t fftResult(LiFFT::types::Idx2D(file.getExtents()[1], file.getExtents()[0]/2+1));
        using FFT_Type = LiFFT::FFT_2D_R2C<TestPrecision>;
        auto input = FFT_Type::wrapInput(file);
        auto output = FFT_Type::wrapOutput(fftResult);
        auto fft = LiFFT::makeFFT<TestLibrary>(input, output);
        file.loadData(true);
        fft(input, output);
        LiFFT::policies::copy(file, baseR2CInput);
        execBaseR2C();
        return checkResult(baseR2COutput, fftResult, "R2C with file input");
    }

    bool testTiffCp(const std::string& filePath)
    {
        std::string filePath2 = filePath+"2.tif";
        tiffWriter::FloatImage<> img(filePath);
        img.saveTo(filePath2);
        img.close();
        tiffWriter::FloatImage<> img1(filePath);
        tiffWriter::FloatImage<> img2(filePath2);
        bool ok = checkResult(img1, img2, "TIFF copy");
        std::remove(filePath2.c_str());
        return ok;
    }

    bool testTiffModify(const std::string& filePath)
    {
        std::string filePath2 = filePath+"2.tif";
        tiffWriter::FloatImage<> img(filePath);
        LiFFT::mem::RealContainer<2, float> data(LiFFT::types::Vec<2>(img.getHeight(), img.getWidth()));
        generateData(data, Circle<float>(50, img.getHeight()/2));
        auto acc = LiFFT::accessors::makeTransposeAccessorFor(img);
        LiFFT::policies::copy(data, img, LiFFT::traits::getIdentityAccessor(data), acc);
        img.saveTo(filePath2);
        img.close();
        tiffWriter::FloatImage<> img2(filePath2);
        auto accData = LiFFT::accessors::makeTransposeAccessorFor(data);
        auto res = compare(data, img2, CmpError(1e-8, 1e-8), accData);
        if(!res.first)
            std::cerr << "Tiff modify failed" << std::endl;
        else
            std::cout << "Tiff modify passed" << std::endl;
        std::remove(filePath2.c_str());
        return res.first;
    }

    bool testTiffFile(const std::string& filePath)
    {
        tiffWriter::FloatImage<> img(filePath, false);
        using FFT_Type = LiFFT::FFT_2D_R2C_F<>;
        auto input = FFT_Type::wrapInput(img);
        auto output = FFT_Type::createNewOutput(input);
        auto fft = LiFFT::makeFFT< TestLibrary, false >(input, output);
        img.load();
        LiFFT::policies::copy(img, baseR2CInput);
        fft(input, output);
        execBaseR2C();
        visualizeOutput(BaseInstance::OutR2C, "outputTiff.pdf");
        return checkResult(baseR2COutput, output, "TIFF test");
    }

    int testFile()
    {
        using FileType = LiFFT::mem::FileContainer<
            tiffWriter::Image<>,
            LiFFT::accessors::ImageAccessorGetColorAsFp<TestPrecision>,
            TestPrecision,
            false
            >;
        FileType myFile("rect.tif");
        TEST( testFile(myFile) );
        TEST( testTiffFile("input1.tif") );
        TEST( testTiffCp("input1.tif") );
        TEST( testTiffModify("input1.tif") );
        return 0;
    }

}  // namespace LiFFTTest
