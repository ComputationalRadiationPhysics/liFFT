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
 
#include "testInplace.hpp"
#include "testUtils.hpp"
#include "haLT/FFT.hpp"
#include "haLT/generateData.hpp"
#include "haLT/types/View.hpp"
#include "haLT/types/SliceView.hpp"
#include <cmath>

using haLT::generateData;
using namespace haLT::generators;

namespace haLTTest {

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
        using FFT = haLT::FFT_3D_R2C_F<true>;
        auto input = FFT::createNewInput(haLT::types::Vec3(2048u, 1024u, 1u));
        haLT::mem::RealContainer<2, float> data(haLT::types::Vec2(2048u, 1024u));
        haLT::mem::RealContainer<3, float> data2(haLT::types::Vec3(2048u, 1024u, 1u));
        auto view = haLT::types::makeSliceView<2>(input, haLT::types::makeRange());
        haLT::generateData(view, GenData());
        haLT::policies::copy(view, data);
        haLT::policies::copy(input, data2);
        bool ok = true;
        for(unsigned z = 0; z < input.getExtents()[2]; z++)
            for(unsigned y = 0; y < input.getExtents()[1]; y++)
            {
                float expected = y + z * input.getExtents()[1];
                float is = data(haLT::types::Vec2(z, y));
                float diff = std::abs(expected - is);
                if(diff > 1e-6)
                    ok = false;
                is = data2(haLT::types::Vec3(z, y, 0u));
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
        using FFT_Type = haLT::FFT_2D_C2C<TestPrecision, true>;
        auto input = FFT_Type::wrapInput(aperture);
        auto output = FFT_Type::createNewOutput(input);
        auto fft = haLT::makeFFT<TestLibrary>(input);
        generateData(input, Rect<TestPrecision>(20,testSize/2));
        haLT::policies::copy(aperture, baseC2CInput);
        fft(input);
        execBaseC2C();
        checkResult(baseC2COutput, output, "C2C inPlace");
    }

    void testInplaceR2C()
    {
        TestExtents ext = TestExtents::all(testSize);
        ext[testNumDims - 1] = (ext[testNumDims - 1] / 2 + 1) * 2;
        using namespace haLT::types;
        auto aperture = makeView(RealContainer(ext), makeRange(Origin(), TestExtents::all(testSize)));
        using FFT_Type = haLT::FFT_2D_R2C<TestPrecision, true>;
        auto input = FFT_Type::wrapInput(aperture);
        auto output = FFT_Type::createNewOutput(input);
        auto fft = haLT::makeFFT<TestLibrary>(input);
        generateData(input, Rect<TestPrecision>(20,testSize/2));
        haLT::policies::copy(aperture, baseR2CInput);
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

}  // namespace haLTTest
