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
 
#include "testInplace.hpp"
#include "testUtils.hpp"
#include "libLiFFT/FFT.hpp"
#include "libLiFFT/generateData.hpp"
#include "libLiFFT/types/View.hpp"
#include "libLiFFT/types/SliceView.hpp"
#include <cmath>

using LiFFT::generateData;
using namespace LiFFT::generators;

namespace LiFFTTest {

    struct GenData
    {

        template< class T_Idx >
        float operator()(T_Idx&& idx) const
        {
            return idx[1] + idx[0]*1024;
        }
    };

    bool testContainerCreation()
    {
        using FFT = LiFFT::FFT_3D_R2C_F<true>;
        auto input = FFT::createNewInput(LiFFT::types::Vec3(2048u, 1024u, 1u));
        LiFFT::mem::RealContainer<2, float> data(LiFFT::types::Vec2(2048u, 1024u));
        LiFFT::mem::RealContainer<3, float> data2(LiFFT::types::Vec3(2048u, 1024u, 1u));
        auto view = LiFFT::types::makeSliceView<2>(input, LiFFT::types::makeRange());
        LiFFT::generateData(view, GenData());
        LiFFT::policies::copy(view, data);
        LiFFT::policies::copy(input, data2);
        bool ok = true;
        for(unsigned z = 0; z < input.getExtents()[2]; z++)
            for(unsigned y = 0; y < input.getExtents()[1]; y++)
            {
                float expected = y + z * input.getExtents()[1];
                float is = data(LiFFT::types::Vec2(z, y));
                float diff = std::abs(expected - is);
                if(diff > 1e-6)
                    ok = false;
                is = data2(LiFFT::types::Vec3(z, y, 0u));
                diff = std::abs(expected - is);
                if(diff > 1e-6)
                    ok = false;
            }
        if(ok)
            std::cout << "Inplace container creation passed" << std::endl;
        else
            std::cerr << "Inplace container creation FAILED" << std::endl;
        return ok;
    }

    bool testInplaceComplex()
    {
        auto aperture = ComplexContainer(TestExtents::all(testSize));
        using FFT_Type = LiFFT::FFT_2D_C2C<TestPrecision, true>;
        auto input = FFT_Type::wrapInput(aperture);
        auto output = FFT_Type::createNewOutput(input);
        auto fft = LiFFT::makeFFT<TestLibrary>(input);
        generateData(input, Rect<TestPrecision>(20,testSize/2));
        LiFFT::policies::copy(aperture, baseC2CInput);
        fft(input);
        execBaseC2C();
        return checkResult(baseC2COutput, output, "C2C inPlace");
    }

    bool testInplaceR2C()
    {
        TestExtents ext = TestExtents::all(testSize);
        ext[testNumDims - 1] = (ext[testNumDims - 1] / 2 + 1) * 2;
        using namespace LiFFT::types;
        auto aperture = makeView(RealContainer(ext), makeRange(Origin(), TestExtents::all(testSize)));
        using FFT_Type = LiFFT::FFT_2D_R2C<TestPrecision, true>;
        auto input = FFT_Type::wrapInput(aperture);
        auto output = FFT_Type::createNewOutput(input);
        auto fft = LiFFT::makeFFT<TestLibrary>(input);
        generateData(input, Rect<TestPrecision>(20,testSize/2));
        LiFFT::policies::copy(aperture, baseR2CInput);
        fft(input);
        execBaseR2C();
        return checkResult(baseR2COutput, output, "R2C inPlace");
    }

    int testInplace()
    {
        TEST( testContainerCreation() );
        TEST( testInplaceComplex() );
        TEST( testInplaceR2C() );
        return 0;
    }

}  // namespace LiFFTTest
