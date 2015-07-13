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
