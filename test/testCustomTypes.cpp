#include "testCustomTypes.hpp"
#include "testUtils.hpp"

#include "foobar/traits/IsComplex.hpp"
#include "foobar/traits/IsBinaryCompatible.hpp"
#include "foobar/traits/IntegralType.hpp"
#include "foobar/types/DimOffsetWrapper.hpp"
#include "foobar/FFT.hpp"
#include "Volume.hpp"
#include "foobar/accessors/VolumeAccessor.hpp"
#include "foobar/libraries/fftw/fftw3Include.h"
#include "generateData.hpp"
#include "foobar/policies/Copy.hpp"
#include "foobar/traits/IdentityAccessor.hpp"
#include <type_traits>
#include <iostream>

namespace foobarTest {

    template< typename T = double >
    struct MyComplex{
        T real, imag;
        MyComplex(){}
        MyComplex(T real): real(real), imag(0){}
        MyComplex(T real, T imag): real(real), imag(imag){}
    };

}  // namespace foobarTest

namespace foobar {
    namespace traits {

        template<typename T>
        struct IsComplex< foobarTest::MyComplex<T> >: std::true_type{};

        template<typename T>
        struct IsBinaryCompatibleImpl< foobarTest::MyComplex<T>, foobar::types::Complex<T> >: std::true_type{};

        template<typename T>
        struct IntegralType< foobarTest::MyComplex<T> >
        {
            using type = T; // or define this in MyComplex itself
        };

        template<>
        struct IsComplex< fftw_complex >: std::true_type{};

        template<>
        struct IsComplex< fftwf_complex >: std::true_type{};

        template<>
        struct IsBinaryCompatibleImpl< fftw_complex, foobar::types::Complex<double> >: std::true_type{};
        template<>
        struct IsBinaryCompatibleImpl< fftwf_complex, foobar::types::Complex<float> >: std::true_type{};
    }  // namespace traits

}  // namespace foobar

namespace foobarTest {

    template<typename T>
    std::ostream& operator<< (std::ostream& stream, MyComplex<T> val){
        stream << val.real << " " << val.imag;
        return stream;
    }

    using ComplexVol     = Volume< MyComplex<TestPrecision> >;
    using ComplexVolFFTW = Volume< std::conditional_t<std::is_same<TestPrecision, float>::value, fftwf_complex, fftw_complex > >;
    using RealVol        = Volume< TestPrecision >;

    using foobar::types::DimOffsetWrapper;
    using ComplexVol2D     = DimOffsetWrapper< ComplexVol,     1 >;
    using ComplexVolFFTW2D = DimOffsetWrapper< ComplexVolFFTW, 1 >;
    using RealVol2D        = DimOffsetWrapper< RealVol,        1 >;

    void testComplex()
    {
        using foobar::accessors::VolumeAccessor;
        ComplexVol2D aperture(testSize, testSize);
        ComplexVol2D fftResult(aperture.xDim(), aperture.yDim(), aperture.zDim());
        using FFT_Type = foobar::FFT_2D_C2C<TestPrecision>;
        auto input = FFT_Type::wrapFFT_Input(aperture, VolumeAccessor());
        auto output = FFT_Type::wrapFFT_Output(fftResult, VolumeAccessor());
        auto fft = foobar::makeFFT<TestLibrary>(input, output);
        generateData(aperture, Rect<TestPrecision>(20,testSize/2), VolumeAccessor());
        fft(input, output);
        foobar::policies::copy(aperture, baseC2CInput, VolumeAccessor());
        execBaseC2C();
        auto res = compare(baseC2COutput, fftResult, CmpError(5e-5, 5e-5), foobar::traits::getIdentityAccessor(baseC2COutput), VolumeAccessor());
        if(!res.first)
            std::cerr << "Error for C2C with custom types: " << res.second << std::endl;
        else
            std::cout << "C2C with custom types passed" << std::endl;
    }

    void testReal()
    {
        using foobar::accessors::VolumeAccessor;
        RealVol2D aperture(testSize, testSize);
        ComplexVolFFTW2D fftResult(aperture.xDim()/2+1, aperture.yDim(), aperture.zDim());
        RealVol2D intensity(aperture.xDim(), aperture.yDim(), aperture.zDim());
        using FFT_Type = foobar::FFT_2D_R2C<TestPrecision>;
        auto input = FFT_Type::wrapFFT_Input(aperture, VolumeAccessor());
        auto output = FFT_Type::wrapFFT_Output(fftResult, VolumeAccessor());
        auto fft = foobar::makeFFT<TestLibrary>(input, output);
        generateData(aperture, Rect<TestPrecision>(20,testSize/2), VolumeAccessor());
        fft(input, output);
        foobar::policies::copy(aperture, baseR2CInput, VolumeAccessor());
        execBaseR2C();
        auto res = compare(baseR2COutput, fftResult, CmpError(5e-5, 5e-5), foobar::traits::getIdentityAccessor(baseR2COutput), VolumeAccessor());
        if(!res.first)
            std::cerr << "Error for R2C with custom types: " << res.second << std::endl;
        else
            std::cout << "R2C with custom types passed" << std::endl;
    }

    void testCustomTypes()
    {
        testComplex();
        testReal();
    }

}  // namespace foobarTest
