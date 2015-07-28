#include "testInplace.hpp"
#include "testUtils.hpp"
#include "foobar/FFT.hpp"
#include "foobar/generateData.hpp"
#include "foobar/types/View.hpp"

using foobar::generateData;
using namespace foobar::generators;

namespace foobarTest {

    void testInplaceComplex()
    {
        auto aperture = ComplexContainer(TestExtents::all(testSize));
        using FFT_Type = foobar::FFT_2D_C2C<TestPrecision, true>;
        auto input = FFT_Type::wrapInput(aperture);
        auto output = FFT_Type::createNewOutput(input);
        auto fft = foobar::makeFFT<TestLibrary>(input);
        generateData(input, Rect<TestPrecision>(20,testSize/2));
        foobar::policies::copy(aperture, baseC2CInput);
        fft(input);
        execBaseC2C();
        checkResult(baseC2COutput, output, "C2C inPlace");
    }

    void testInplaceR2C()
    {
        TestExtents ext = TestExtents::all(testSize);
        ext[testNumDims - 1] = (ext[testNumDims - 1] / 2 + 1) * 2;
        using namespace foobar::types;
        auto aperture = makeView(RealContainer(ext), makeRange(Origin(), TestExtents::all(testSize)));
        using FFT_Type = foobar::FFT_2D_R2C<TestPrecision, true>;
        auto input = FFT_Type::wrapInput(aperture);
        auto output = FFT_Type::createNewOutput(input);
        auto fft = foobar::makeFFT<TestLibrary>(input);
        generateData(input, Rect<TestPrecision>(20,testSize/2));
        foobar::policies::copy(aperture, baseR2CInput);
        fft(input);
        execBaseR2C();
        checkResult(baseR2COutput, output, "R2C inPlace");
    }

    void testInplace()
    {
        testInplaceComplex();
        testInplaceR2C();
    }

}  // namespace foobarTest
