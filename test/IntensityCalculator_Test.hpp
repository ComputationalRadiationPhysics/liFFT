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

#include "haLT/policies/IntensityCalculator.hpp"
#include "haLT/policies/GetNumElements.hpp"

void testIntensityCalculator()
{
    using namespace haLT::types;
    using namespace haLT::mem;
    using haLT::policies::getNumElements;
    using haLT::policies::useIntensityCalculator;
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
    complexSoA.data.getRealData() = new Real<double>[getNumElements(complexSoA)];
    complexSoA.data.getImagData() = new Real<double>[getNumElements(complexSoA)];

    real3D.extents = {NumVals, NumVals+1, NumVals+2};
    real3D.data = new Real<double>[getNumElements(real3D)];

    complexAoS3D.extents = {NumVals, NumVals+1, NumVals+2};
    complexAoS3D.data = new Complex<double>[getNumElements(complexAoS3D)];

    complexSoA3D.extents = {NumVals, NumVals+1, NumVals+2};
    complexSoA3D.data.getRealData() = new Real<double>[getNumElements(complexSoA3D)];
    complexSoA3D.data.getImagData() = new Real<double>[getNumElements(complexSoA3D)];

    useIntensityCalculator(real, result.data());
    useIntensityCalculator(complexAoS, result.data());
    useIntensityCalculator(complexSoA, result.data());
    useIntensityCalculator(real3D, result.data());
    useIntensityCalculator(complexAoS3D, result.data());
    useIntensityCalculator(complexSoA3D, result.data());
}
