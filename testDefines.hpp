#pragma once

#include "foobar/mem/DataContainer.hpp"
#include "foobar/mem/RealValues.hpp"
#include "foobar/mem/ComplexAoSValues.hpp"

// Types used for the test suite
constexpr unsigned testNumDims = 2;
using TestExtents      = foobar::types::Vec<testNumDims>;
using TestPrecision    = float;
using RealType         = foobar::mem::RealValues<TestPrecision>;
using ComplexType      = foobar::mem::ComplexAoSValues<TestPrecision>;
using RealContainer    = foobar::mem::DataContainer<testNumDims, RealType>;
using ComplexContainer = foobar::mem::DataContainer<testNumDims, ComplexType>;
using TestR2CInput  = RealContainer;
using TestR2COutput = ComplexContainer;
using TestC2CInput  = ComplexContainer;
using TestC2COutput = ComplexContainer;

// Control values used in the test suite
extern TestR2CInput  testR2CInput;
extern TestR2COutput testR2COutput;
extern TestC2CInput  testC2CInput;
extern TestC2COutput testC2COutput;
