#pragma once

#include "foobar/types/Real.hpp"

namespace foobar {
namespace types {

    /**
     * Type used to store complex values (real and imaginary part)
     * Uses the template parameter to choose the memory type (precision)
     */
    template< typename T=double >
    struct Complex
    {
        using type = T;
        static constexpr bool isComplex = true;
        Real<T> real, imag;

        Complex(){}
        template< typename U, typename V = T >
        Complex(U&& real, V&& imag = V(0)): real(std::forward<U>(real)), imag(std::forward<V>(imag)){}
    };

    /**
     * Generic reference to a complex value
     * Can be used with either AoS or SoA
     */
    template< typename T=double >
    struct ComplexRef
    {
        using type = T;
        static constexpr bool isComplex = true;
        using Real_t = Real<T>;
        using Complex_t = Complex<T>;

        Real_t &real, &imag;

        explicit ComplexRef(Complex_t& value): real(value.real), imag(value.imag){}
        ComplexRef(Real_t& r, Real_t& i): real(r), imag(i){}

        ComplexRef&
        operator=(const Complex_t& c){
            real = c.real;
            imag = c.imag;
            return *this;
        }
    };

}  // namespace types
}  // namespace foobar
