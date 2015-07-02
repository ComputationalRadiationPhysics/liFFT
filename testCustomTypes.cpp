#include "testCustomTypes.hpp"
#include "testUtils.hpp"

#include "foobar/traits/IsComplex.hpp"
#include "foobar/traits/IsBinaryCompatible.hpp"
#include "foobar/traits/IntegralType.hpp"
#include "foobar/types/DimOffsetWrapper.hpp"
#include "foobar/FFT.hpp"
#include "Volume.hpp"
#include "foobar/policies/VolumeAccessor.hpp"
#include "foobar/libraries/fftw/fftw3Include.h"
#include "generateData.hpp"
#include <type_traits>

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

    }  // namespace traits

}  // namespace foobar

namespace foobarTest {

    template<typename T>
    std::ostream& operator<< (std::ostream& stream, MyComplex<T> val){
        stream << val.real << " " << val.imag;
        return stream;
    }

    using ComplexVol     = Volume< MyComplex<> >;
    using ComplexVolFFTW = Volume< fftw_complex >;
    using RealVol        = Volume<>;

    using foobar::types::DimOffsetWrapper;
    using ComplexVol2D     = DimOffsetWrapper< ComplexVol,     1 >;
    using ComplexVolFFTW2D = DimOffsetWrapper< ComplexVolFFTW, 1 >;
    using RealVol2D        = DimOffsetWrapper< RealVol,        1 >;

    void testComplex()
    {
        ComplexVol2D aperture(1024, 1024);
        ComplexVol2D fftResult(aperture.xDim(), aperture.yDim(), aperture.zDim());
        using FFT_Type = foobar::FFT_2D_C2C_D;
        auto input = FFT_Type::wrapFFT_Input(aperture, foobar::policies::VolumeAccessor());
        auto output = FFT_Type::wrapFFT_Output(fftResult, foobar::policies::VolumeAccessor());
        auto fft = foobar::makeFFT<foobar::libraries::fftw::FFTW<>>(input, output);
        generateData(aperture, Rect<double>(20,20,500,500), foobar::policies::VolumeAccessor());
        fft(input, output);
    }

    void testReal()
    {
        RealVol2D aperture(1024, 1024);
        ComplexVolFFTW2D fftResult(aperture.xDim()/2+1, aperture.yDim(), aperture.zDim());
        RealVol2D intensity(aperture.xDim(), aperture.yDim(), aperture.zDim());
        using FFT_Type = foobar::FFT_2D_R2C_D;
        auto input = FFT_Type::wrapFFT_Input(aperture, foobar::policies::VolumeAccessor());
        auto output = FFT_Type::wrapFFT_Output(fftResult, foobar::policies::VolumeAccessor());
        auto fft = foobar::makeFFT<foobar::libraries::fftw::FFTW<>>(input, output);
        generateData(aperture, Rect<double>(20,20,500,500), foobar::policies::VolumeAccessor());
        fft(input, output);
    }

    void testCustomTypes()
    {
        testComplex();
        testReal();
    }

}  // namespace foobarTest
