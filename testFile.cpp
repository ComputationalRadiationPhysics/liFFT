#include "testFile.hpp"
#include "testUtils.hpp"

#include "foobar/mem/DataContainer.hpp"
#include "foobar/mem/FileContainer.hpp"
#include "foobar/policies/ImageAccessor.hpp"
#include "libTiff/libTiff.hpp"
#include "libTiff/traitsAndPolicies.hpp"
#include "foobar/FFT.hpp"

namespace foobarTest {

    template< typename T_File >
    void testFile( T_File& file )
    {
        using FFTResult_t = foobar::mem::ComplexContainer<2, float>;
        FFTResult_t fftResult;
        fftResult.extents = foobar::types::Idx2D(file.getExtents()[1], file.getExtents()[0]/2+1);
        fftResult.allocData();
        using FFT_Type = foobar::FFT_2D_R2C_F;
        auto input = FFT_Type::wrapFFT_Input(file);
        auto output = FFT_Type::wrapFFT_Output(fftResult);
        auto fft = foobar::makeFFT<TestLibrary>(input, output);
        file.loadData(true);
        fft(input, output);
    }

    void testFile()
    {
        using FileType = foobar::mem::FileContainer<
            libTiff::Image<>,
            foobar::policies::ImageAccessorGetColorAsFp<>,
            float,
            false
            >;
        FileType myFile("rect.tif");
        testFile(myFile);
    }

}  // namespace foobarTest
