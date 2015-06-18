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
        template< typename U, typename = std::enable_if_t< std::is_integral<U>::value > >
        Complex(U&& real): real(std::forward<U>(real)), imag(0){}
        template< typename U, typename V >
        Complex(U&& real, V&& imag): real(std::forward<U>(real)), imag(std::forward<V>(imag)){}

        template< typename U >
        Complex&
        operator=(U&& real)
        {
            real = std::forward<U>(real);
            imag = 0;
            return *this;
        }
    };

    /**
     * Generic reference to a complex value
     * Can be used with either AoS or SoA
     */
    template< typename T=double, bool T_isConst = false >
    struct ComplexRef
    {
        using type = T;
        static constexpr bool isConst = true;
        static constexpr bool isComplex = true;
        using Real_t = std::conditional_t< isConst, const Real<T>, Real<T> >;
        using Complex_t = std::conditional_t< isConst, const Complex<T>, Complex<T> >;

        Real_t &real;
        Real_t &imag;

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

namespace std {

    template< typename T >
    struct is_lvalue_reference< foobar::types::ComplexRef<T> >: std::true_type{};

}  // namespace std
