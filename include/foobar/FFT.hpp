#pragma once

#include "foobar/FFT_Properties.hpp"
#include "foobar/FFT_Impl.hpp"

// Included for convenience, so only one include is required from user code
#include "foobar/FFT_Definition.hpp"
#include "foobar/FFT_DataWrapper.hpp"

namespace foobar {

    /**
     * Assembles an FFT
     *
     * Usage:
     *      1) The constructor takes the container(s) and may modify the memory returned by the GetRawPtr policy
     *         Note: An implementation may not need the memory at all
     *      2) Execute the FFT with <fftInstance>(input, output), which calls the ReadData<Input> policy, than performs the transform
     *         from the memories returned by the GetRawPtr policies and calls the WriteData<Output> policy afterwards
     *
     * Parameters:
     * \tparam T_Library FFT Library to use
     * \tparam T_InputWrapper   Input wrapped in a FFT_DataWrapper
     * \tparam T_OutputWrapper  Output wrapped in a FFT_DataWrapper
     * \tparam T_constructWithReadOnly If true, the data passed in the constructor is not overwritten. Use false for better performance
     */
    template<
            class T_Library,
            typename T_InputWrapper,
            typename T_OutputWrapper,
            bool T_constructWithReadOnly = true
            >
    class FFT
    {
        using Library = T_Library;
        using Input = T_InputWrapper;
        using Output = T_OutputWrapper;
        static constexpr bool constructWithReadOnly = T_constructWithReadOnly;

        using FFT_Def = typename Input::FFT_Def;
        static_assert(std::is_same< FFT_Def, typename Output::FFT_Def>::value, "FFT types of input and output differs");
        using FFT_Properties = detail::FFT_Properties< FFT_Def, Input, Output, constructWithReadOnly >;
        using ActLibrary = typename bmpl::apply< Library, FFT_Properties >::type;
        static constexpr bool isInplace = FFT_Properties::isInplace;

        ActLibrary lib_;
    public:
        explicit FFT(Input& input, Output& output): lib_(input, output)
        {
            static_assert(!isInplace, "Must not be called for inplace transforms");
        }

        explicit FFT(Input& inOut): lib_(inOut)
        {
            static_assert(isInplace, "Must not be called for out-of-place transforms");
        }

        FFT(FFT&& obj): lib_(std::move(obj.lib_)){}

        void operator()(Input& input, Output& output)
        {
            static_assert(!isInplace, "Must not be called for inplace transforms");
            input.preProcess();
            lib_(input, output);
            output.postProcess();
        }

        void operator()(Input& inout)
        {
            static_assert(isInplace, "Must not be called for out-of-place transforms");
            inout.preProcess();
            lib_(inout);
            inout.postProcess();
        }
    };

    template<
        class T_Library,
        bool T_constructWithReadOnly = true,
        typename T_InputWrapper,
        typename T_OutputWrapper
        >
    FFT< T_Library, std::decay_t<T_InputWrapper>, std::decay_t<T_OutputWrapper>, T_constructWithReadOnly >
    makeFFT(T_InputWrapper&& input, T_OutputWrapper&& output)
    {
        return FFT< T_Library, std::decay_t<T_InputWrapper>, std::decay_t<T_OutputWrapper>, T_constructWithReadOnly >(input, output);
    }

}  // namespace foobar
