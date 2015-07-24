#include "testInplace.hpp"
#include "testUtils.hpp"
#include "foobar/FFT.hpp"
#include "generateData.hpp"

namespace foobarTest {

    void testInplaceComplex()
    {
        auto aperture = ComplexContainer(TestExtents::all(testSize));
        using FFT_Type = foobar::FFT_2D_C2C<TestPrecision, true>;
        auto input = FFT_Type::wrapFFT_Input(aperture);
        auto output = FFT_Type::getNewFFT_Output(input);
        auto fft = foobar::makeFFT<TestLibrary>(input);
        generateData(input, Rect<TestPrecision>(20,testSize/2));
        foobar::policies::copy(aperture, baseC2CInput);
        fft(input);
        execBaseC2C();
        checkResult(baseC2COutput, output, "C2C inPlace");
    }

    void testInplace()
    {
        testInplaceComplex();
    }

}  // namespace foobarTest
