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

#include "haLT/mem/DataContainer.hpp"
#include "haLT/mem/RealValues.hpp"
#include "haLT/mem/ComplexAoSValues.hpp"
#ifdef WITH_CUDA
#include "haLT/libraries/cuFFT/cuFFT.hpp"
#else
#include "haLT/libraries/fftw/FFTW.hpp"
#endif

namespace haLTTest{
    // Types used for the test suite
    constexpr unsigned testNumDims = 2;
    constexpr unsigned testSize = 2048;
    using TestExtents      = haLT::types::Vec<testNumDims>;
    using TestPrecision    = float;
    using RealType         = haLT::mem::RealValues<TestPrecision>;
    using ComplexType      = haLT::mem::ComplexAoSValues<TestPrecision>;
    using RealContainer    = haLT::mem::DataContainer<testNumDims, RealType>;
    using ComplexContainer = haLT::mem::DataContainer<testNumDims, ComplexType>;
    using BaseR2CInput  = RealContainer;
    using BaseR2COutput = ComplexContainer;
    using BaseC2CInput  = ComplexContainer;
    using BaseC2COutput = ComplexContainer;

    #ifdef WITH_CUDA
    using TestLibrary = haLT::libraries::cuFFT::CuFFT<>;
    #else
    using TestLibrary = haLT::libraries::fftw::FFTW<>;
    #endif

    // Control values used in the test suite
    extern BaseR2CInput  baseR2CInput;
    extern BaseR2COutput baseR2COutput;
    extern BaseC2CInput  baseC2CInput;
    extern BaseC2COutput baseC2COutput;
}
