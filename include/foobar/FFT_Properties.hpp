#pragma once

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
        using Input = T_Input;
        using Output = T_Output;
        static constexpr bool constructWithReadOnly = T_constructWithReadOnly;

        static constexpr bool isComplexIn = Input::isComplex;
        static constexpr bool isAoSIn = Input::isAoS;
        static constexpr bool isStridedIn = Input::isStrided;

        static constexpr bool isComplexOut = Output::isComplex;
        static constexpr bool isAoSOut = Output::isAoS;
        static constexpr bool isStridedOut = Output::isStrided;

        static constexpr bool isAoS = isAoSIn || isAoSOut;
        static constexpr bool isStrided = isStridedIn || isStridedOut;
    };

}  // namespace detail
}  // namespace foobar
