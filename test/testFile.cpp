#include "testFile.hpp"
#include "testUtils.hpp"

#include "foobar/mem/DataContainer.hpp"
#include "foobar/mem/FileContainer.hpp"
#include "foobar/accessors/ImageAccessor.hpp"
#include "tiffWriter/image.hpp"
#include "tiffWriter/traitsAndPolicies.hpp"
#include "foobar/FFT.hpp"
#include "foobar/generateData.hpp"
#include "foobar/accessors/TransposeAccessor.hpp"

using foobar::generateData;
using namespace foobar::generators;

namespace foobarTest {

    template< typename T_File >
    void testFile( T_File& file )
    {
        using FFTResult_t = foobar::mem::ComplexContainer<2, TestPrecision>;
        FFTResult_t fftResult(foobar::types::Idx2D(file.getExtents()[1], file.getExtents()[0]/2+1));
        using FFT_Type = foobar::FFT_2D_R2C<TestPrecision>;
        auto input = FFT_Type::wrapInput(file);
        auto output = FFT_Type::wrapOutput(fftResult);
        auto fft = foobar::makeFFT<TestLibrary>(input, output);
        file.loadData(true);
        fft(input, output);
        foobar::policies::copy(file, baseR2CInput);
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
        foobar::mem::RealContainer<2, float> data(foobar::types::Vec<2>(img.getHeight(), img.getWidth()));
        generateData(data, Circle<float>(50, img.getHeight()/2));
        auto acc = foobar::accessors::makeTransposeAccessorFor(img);
        foobar::policies::copy(data, img, foobar::traits::getIdentityAccessor(data), acc);
        img.saveTo(filePath2);
        img.close();
        tiffWriter::FloatImage<> img2(filePath2);
        auto accData = foobar::accessors::makeTransposeAccessorFor(data);
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
        using FFT_Type = foobar::FFT_2D_R2C_F<>;
        auto input = FFT_Type::wrapInput(img);
        auto output = FFT_Type::createNewOutput(input);
        auto fft = foobar::makeFFT< TestLibrary, false >(input, output);
        img.load();
        foobar::policies::copy(img, baseR2CInput);
        fft(input, output);
        execBaseR2C();
        checkResult(baseR2COutput, output, "TIFF test");
        visualizeOutput(BaseInstance::OutR2C, "outputTiff.pdf");
    }

    void testFile()
    {
        using FileType = foobar::mem::FileContainer<
            tiffWriter::Image<>,
            foobar::accessors::ImageAccessorGetColorAsFp<TestPrecision>,
            TestPrecision,
            false
            >;
        FileType myFile("rect.tif");
        testFile(myFile);
        testTiffFile("input1.tif");
        testTiffCp("input1.tif");
        testTiffModify("input1.tif");
    }

}  // namespace foobarTest
