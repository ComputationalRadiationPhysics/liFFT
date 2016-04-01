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
 
#pragma once

#include "foobar/mem/DataContainer.hpp"
#include "foobar/mem/RealValues.hpp"
#include "foobar/mem/ComplexAoSValues.hpp"
#ifdef WITH_CUDA
#include "foobar/libraries/cuFFT/cuFFT.hpp"
#else
#include "foobar/libraries/fftw/FFTW.hpp"
#endif

namespace foobarTest{
    // Types used for the test suite
    constexpr unsigned testNumDims = 2;
    constexpr unsigned testSize = 2048;
    using TestExtents      = foobar::types::Vec<testNumDims>;
    using TestPrecision    = float;
    using RealType         = foobar::mem::RealValues<TestPrecision>;
    using ComplexType      = foobar::mem::ComplexAoSValues<TestPrecision>;
    using RealContainer    = foobar::mem::DataContainer<testNumDims, RealType>;
    using ComplexContainer = foobar::mem::DataContainer<testNumDims, ComplexType>;
    using BaseR2CInput  = RealContainer;
    using BaseR2COutput = ComplexContainer;
    using BaseC2CInput  = ComplexContainer;
    using BaseC2COutput = ComplexContainer;

    #ifdef WITH_CUDA
    using TestLibrary = foobar::libraries::cuFFT::CuFFT<>;
    #else
    using TestLibrary = foobar::libraries::fftw::FFTW<>;
    #endif

    // Control values used in the test suite
    extern BaseR2CInput  baseR2CInput;
    extern BaseR2COutput baseR2COutput;
    extern BaseC2CInput  baseC2CInput;
    extern BaseC2COutput baseC2COutput;
}
