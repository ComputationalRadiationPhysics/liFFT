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
 
#include "testCustomTypes.hpp"
#include "testUtils.hpp"

#include "haLT/traits/IsComplex.hpp"
#include "haLT/traits/IsBinaryCompatible.hpp"
#include "haLT/traits/IntegralType.hpp"
#include "haLT/FFT.hpp"
#include "Volume.hpp"
#include "haLT/accessors/VolumeAccessor.hpp"
#include "haLT/libraries/fftw/fftw3Include.h"
#include "haLT/generateData.hpp"
#include "haLT/policies/Copy.hpp"
#include "haLT/traits/IdentityAccessor.hpp"
#include "haLT/types/SliceView.hpp"
#include <type_traits>
#include <iostream>

using haLT::generateData;
using namespace haLT::generators;

namespace haLTTest {

    template< typename T = double >
    struct MyComplex{
        T real, imag;
        MyComplex(){}
        MyComplex(T realIn): real(realIn), imag(0){}
        MyComplex(T realIn, T imagIn): real(realIn), imag(imagIn){}
    };

}  // namespace haLTTest

namespace haLT {
    namespace traits {

        template<typename T>
        struct IsComplex< haLTTest::MyComplex<T> >: std::true_type{};

        template<typename T>
        struct IsBinaryCompatibleImpl< haLTTest::MyComplex<T>, haLT::types::Complex<T> >: std::true_type{};

        template<typename T>
        struct IntegralType< haLTTest::MyComplex<T> >
        {
            using type = T; // or define this in MyComplex itself
        };

        template<>
        struct IsComplex< fftw_complex >: std::true_type{};

        template<>
        struct IsComplex< fftwf_complex >: std::true_type{};

        template<>
        struct IsBinaryCompatibleImpl< fftw_complex, haLT::types::Complex<double> >: std::true_type{};
        template<>
        struct IsBinaryCompatibleImpl< fftwf_complex, haLT::types::Complex<float> >: std::true_type{};

        template<typename T>
        struct IdentityAccessor< Volume<T> >
        {
            using type = accessors::VolumeAccessor;
        };
    }  // namespace traits

}  // namespace haLT

namespace haLTTest {

    template<typename T>
    std::ostream& operator<< (std::ostream& stream, MyComplex<T> val){
        stream << val.real << " " << val.imag;
        return stream;
    }

    using ComplexVol     = Volume< MyComplex<TestPrecision> >;
    using ComplexVolFFTW = Volume< std::conditional_t<std::is_same<TestPrecision, float>::value, fftwf_complex, fftw_complex > >;
    using RealVol        = Volume< TestPrecision >;

    using haLT::types::makeRange;
    using haLT::types::makeSliceView;

    void testComplex()
    {
        using haLT::accessors::VolumeAccessor;
        auto aperture = ComplexVol(testSize, testSize);
        auto fftResult = ComplexVol(aperture.xDim(), aperture.yDim(), aperture.zDim());
        using FFT_Type = haLT::FFT_2D_C2C<TestPrecision>;
        auto input = FFT_Type::wrapInput(makeSliceView<0>(aperture));
        auto output = FFT_Type::wrapOutput(makeSliceView<0>(fftResult));
        auto fft = haLT::makeFFT<TestLibrary>(input, output);
        generateData(aperture, Rect<TestPrecision>(20,testSize/2));
        fft(input, output);
        haLT::policies::copy(makeSliceView<0>(aperture), baseC2CInput);
        execBaseC2C();
        checkResult(baseC2COutput, makeSliceView<0>(fftResult), "C2C with custom types");
    }

    void testReal()
    {
        using haLT::accessors::VolumeAccessor;
        auto aperture = RealVol(testSize, testSize);
        auto fftResult = ComplexVolFFTW(aperture.xDim()/2+1, aperture.yDim(), aperture.zDim());
        using FFT_Type = haLT::FFT_2D_R2C<TestPrecision>;
        auto input = FFT_Type::wrapInput(makeSliceView<0>(aperture));
        auto output = FFT_Type::wrapOutput(makeSliceView<0>(fftResult));
        auto fft = haLT::makeFFT<TestLibrary>(input, output);
        generateData(aperture, Rect<TestPrecision>(20,testSize/2));
        fft(input, output);
        haLT::policies::copy(makeSliceView<0>(aperture), baseR2CInput);
        execBaseR2C();
        checkResult(baseR2COutput, makeSliceView<0>(fftResult), "R2C with custom types");
    }

    void testCustomTypes()
    {
        testComplex();
        testReal();
    }

}  // namespace haLTTest
