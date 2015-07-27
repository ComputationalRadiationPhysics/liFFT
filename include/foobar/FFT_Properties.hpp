#pragma once

#include "foobar/FFT_Kind.hpp"

namespace foobar {
namespace detail {

    /**
     * Type that is passed to FFT-Implementations to define some properties common for all FFTs
     *
     * \tparam T_FFT_Def Specialized FFT_Definition used in CRTP to make its definitions available
     * \tparam T_Input Input wrapper
     * \tparam T_Output Output wrapper
     */
    template<
        typename T_FFT_Def,
        typename T_Input,
        typename T_Output,
        bool T_constructWithReadOnly
        >
    struct FFT_Properties: public T_FFT_Def
    {
    public:
        using FFT_Def = T_FFT_Def;
        using Input = T_Input;
        using Output = T_Output;
        static constexpr bool constructWithReadOnly = T_constructWithReadOnly;

        static constexpr bool isComplexIn = Input::isComplex;
        static constexpr bool isAoSIn = Input::isAoS;
        static constexpr bool isStridedIn = Input::isStrided;

        static constexpr bool isComplexOut = (!FFT_Def::isInplace && Output::isComplex) ||
                                             ( FFT_Def::isInplace && FFT_Def::kind != FFT_Kind::Complex2Real);
        static constexpr bool isAoSOut = Output::isAoS;
        static constexpr bool isStridedOut = Output::isStrided;

        static constexpr bool isAoS = isAoSIn || isAoSOut;
        static constexpr bool isStrided = isStridedIn || isStridedOut;
    };

}  // namespace detail
}  // namespace foobar
