#pragma once

#include <foobar/FFT_Properties.hpp>
#include "foobar/AutoDetect.hpp"
#include "foobar/types/InplaceType.hpp"
#include "foobar/FFT_Properties.hpp"
#include "foobar/FFT_Impl.hpp"

namespace foobar {

    /**
     * Assembles an FFT class that can be queried with the type member
     *
     * Usage:
     *      1) The constructor takes the container(s) and may modify the memory returned by the GetRawPtr policy
     *         Note: An implementation may not need the memory at all
     *      2) Execute the FFT with <fftInstance>(input, output), which calls the ReadData<Input> policy, than performs the transform
     *         from the memories returned by the GetRawPtr policies and calls the WriteData<Output> policy afterwards
     *
     * Parameters:
     * @param T_Library FFT Library to use
     * @param T_Input   Input container type
     * @param T_Output  Output container type (Can be an InplaceType for in-place transforms)
     * @param T_IsFwd   Whether to use forward or backward transform (should have a bool ::value member) or AutoDetect (True for C2C or R2C, False for C2R)
     */
    template<
            class T_Library,
            typename T_Input,
            typename T_Output = types::InplaceType<>,
            typename T_IsFwd = AutoDetect
            >
    struct FFT
    {
        using Library = T_Library;
        using Input = T_Input;
        using Output = T_Output;
        using IsFwd = T_IsFwd;

        using FFT_Properties = detail::FFT_Properties< Input, Output, IsFwd >;

        using type = detail::FFT_Impl< Library, FFT_Properties >;
    };

}  // namespace foobar
