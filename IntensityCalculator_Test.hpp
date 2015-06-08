#pragma once

#include "foobar/policies/IntensityCalculator.hpp"
#include "foobar/policies/GetNumElements.hpp"
#include "foobar/types/all.hpp"

void testIntensityCalculator()
{
    using namespace foobar::types;
    using foobar::policies::getNumElements;
    using foobar::policies::useIntensityCalculator;
    DataContainer< 1, RealValues<double>> real;
    DataContainer< 1, ComplexAoSValues<double>> complexAoS;
    DataContainer< 1, ComplexSoAValues<double>> complexSoA;
    DataContainer< 3, RealValues<double>> real3D;
    DataContainer< 3, ComplexAoSValues<double>> complexAoS3D;
    DataContainer< 3, ComplexSoAValues<double>> complexSoA3D;

    std::vector<double> result(NumVals);

    real.extents = {NumVals};
    real.data = new Real<double>[getNumElements(real)];

    complexAoS.extents = {NumVals};
    complexAoS.data = new Complex<double>[getNumElements(complexAoS)];

    complexSoA.extents = {NumVals};
    complexSoA.data.real = new Real<double>[getNumElements(complexSoA)];
    complexSoA.data.imag = new Real<double>[getNumElements(complexSoA)];

    real3D.extents = {NumVals, NumVals+1, NumVals+2};
    real3D.data = new Real<double>[getNumElements(real3D)];

    complexAoS3D.extents = {NumVals, NumVals+1, NumVals+2};
    complexAoS3D.data = new Complex<double>[getNumElements(complexAoS3D)];

    complexSoA3D.extents = {NumVals, NumVals+1, NumVals+2};
    complexSoA3D.data.real = new Real<double>[getNumElements(complexSoA3D)];
    complexSoA3D.data.imag = new Real<double>[getNumElements(complexSoA3D)];

    useIntensityCalculator(real, result.data());
    useIntensityCalculator(complexAoS, result.data());
    useIntensityCalculator(complexSoA, result.data());
    useIntensityCalculator(real3D, result.data());
    useIntensityCalculator(complexAoS3D, result.data());
    useIntensityCalculator(complexSoA3D, result.data());
}
