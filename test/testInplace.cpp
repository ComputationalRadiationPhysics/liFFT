#include "testInplace.hpp"
#include "testUtils.hpp"
#include "foobar/FFT.hpp"
#include "foobar/generateData.hpp"
#include "foobar/types/View.hpp"
#include "foobar/types/SliceView.hpp"
#include <cmath>

using foobar::generateData;
using namespace foobar::generators;

namespace foobarTest {

    struct GenData
    {

        template< class T_Idx >
        float operator()(T_Idx&& idx) const
        {
            return idx[1] + idx[0]*1024;
        }
    };

    void
    testContainerCreation()
    {
        using FFT = foobar::FFT_3D_R2C_F<true>;
        auto input = FFT::createNewInput(foobar::types::Vec3(2048u, 1024u, 1u));
        foobar::mem::RealContainer<2, float> data(foobar::types::Vec2(2048u, 1024u));
        foobar::mem::RealContainer<3, float> data2(foobar::types::Vec3(2048u, 1024u, 1u));
        auto view = foobar::types::makeSliceView<2>(input, foobar::types::makeRange());
        foobar::generateData(view, GenData());
        foobar::policies::copy(view, data);
        foobar::policies::copy(input, data2);
        bool ok = true;
        for(unsigned z = 0; z < input.getExtents()[2]; z++)
            for(unsigned y = 0; y < input.getExtents()[1]; y++)
            {
                float expected = y + z * input.getExtents()[1];
                float is = data(foobar::types::Vec2(z, y));
                float diff = std::abs(expected - is);
                if(diff > 1e-6)
                    ok = false;
                is = data2(foobar::types::Vec3(z, y, 0u));
                diff = std::abs(expected - is);
                if(diff > 1e-6)
                    ok = false;
            }
        if(ok)
            std::cout << "Inplace container creation passed" << std::endl;
        else
            std::cerr << "Inplace container creation FAILED" << std::endl;
        exit(0);
    }

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
        testContainerCreation();
        testInplaceComplex();
        testInplaceR2C();
    }

}  // namespace foobarTest
