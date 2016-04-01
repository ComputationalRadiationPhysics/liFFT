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
 
#include "testPlainPtr.hpp"
#include "testUtils.hpp"
#include "haLT/types/Complex.hpp"
#include "haLT/types/Real.hpp"
#include "haLT/mem/PlainPtrWrapper.hpp"
#include "haLT/FFT.hpp"
#include "haLT/policies/Copy.hpp"
#include <iostream>

namespace haLTTest {

    void testPlainPtr()
    {
        static_assert(testNumDims == 2, "Currently only 2D is tested");
        using Real = haLT::types::Real<TestPrecision>;
        using Complex = haLT::types::Complex<TestPrecision>;
        std::unique_ptr<Real[]> input(new Real[testSize*testSize]);
        std::unique_ptr<Complex[]> output(new Complex[testSize*(testSize/2+1)]);
        for(unsigned i=0; i<testSize*testSize; ++i)
            input[i] = std::rand() / RAND_MAX;
        using FFT_TYPE = haLT::FFT_2D_R2C<TestPrecision>;
        auto inWrapped = FFT_TYPE::wrapInput( haLT::mem::wrapPtr<false>(input.get(), TestExtents(testSize, testSize)) );
        auto outWrapped = FFT_TYPE::wrapOutput(haLT::mem::wrapPtr<true>(output.get(), TestExtents(testSize, testSize/2+1)));
        auto fft = haLT::makeFFT<TestLibrary>(inWrapped, outWrapped);
        fft(inWrapped, outWrapped);
        haLT::policies::copy(inWrapped, baseR2CInput);
        execBaseR2C();
        checkResult(baseR2COutput, outWrapped, "R2C with PlainPtrWrapper");
    }

}  // namespace haLTTest
