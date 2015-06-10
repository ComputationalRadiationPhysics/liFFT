#pragma once

#include <boost/mpl/apply.hpp>
#include "foobar/traits/IntegralType.hpp"
#include "foobar/traits/IsComplex.hpp"
#include "foobar/traits/IsAoS.hpp"
#include "foobar/traits/IsStrided.hpp"
#include "foobar/traits/NumDims.hpp"
#include "foobar/types/InplaceType.hpp"
#include "foobar/policies/ReadData.hpp"
#include "foobar/policies/WriteData.hpp"
#include "util.hpp"

namespace bmpl = boost::mpl;

namespace foobar {

    /**
     * Type used to indicate that a given value should be automatically detected
     */
    struct AutoDetect: std::integral_constant<unsigned, 0>{};

namespace policies {

    namespace detail {

        /**
         * Type that is passed to FFT-Implementations to define some properties common for all FFTs
         * Parameters are the same as for FFT itself minus the Library
         */
        template<
            typename T_Input,
            typename T_Output,
            typename T_IsFwd,
            typename T_NumDims
            >
        struct FFT_Properties
        {
        public:
            using Input = T_Input;
            static constexpr bool isInplace = traits::IsInplace<T_Output>::value;
            using Output = typename std::conditional< isInplace, Input, T_Output >::type;
        private:
            static constexpr bool autoDetectIsFwd = std::is_same< T_IsFwd, AutoDetect >::value;
            static constexpr bool autoDetectNumDims = std::is_same< T_NumDims, AutoDetect >::value;
            static_assert(autoDetectIsFwd || std::is_same< typename T_IsFwd::value_type, bool >::value, "Wrong argument type for IsFwd");
            static_assert(autoDetectNumDims || std::is_same< typename T_NumDims::value_type, unsigned >::value, "Wrong argument type for NumDims");

            using PrecisionTypeIn = typename traits::IntegralType<Input>::type;
            using NumDimsIn = typename std::conditional<
                                autoDetectNumDims,
                                traits::NumDims<Input>,
                                T_NumDims >::type;
            using PrecisionTypeOut = typename traits::IntegralType<Output>::type;
            using NumDimsOut = typename std::conditional<
                                 autoDetectNumDims,
                                 traits::NumDims<Output>,
                                 T_NumDims >::type;

            static_assert( AssertValue< std::is_same< PrecisionTypeIn, PrecisionTypeOut > >::value, "Need same precision on In/Out");
            static_assert(NumDimsIn::value >= 1, "Need >= 1 dimension");
            static_assert(NumDimsOut::value >= 1, "Need >= 1 dimension");
            static_assert(NumDimsIn::value == NumDimsOut::value, "Dimension mismatch");
        public:
            static constexpr bool isComplexIn = traits::IsComplex<Input>::value;
            static constexpr bool isAoSIn = traits::IsAoS<Input>::value;
            static constexpr bool isStridedIn = traits::IsStrided<Input>::value;

            static constexpr bool isComplexOut = traits::IsComplex< typename std::conditional< isInplace, T_Output, Output >::type >::value;
            static constexpr bool isAoSOut = traits::IsAoS<Output>::value;
            static constexpr bool isStridedOut = traits::IsStrided<Output>::value;

            using PrecisionType = PrecisionTypeIn;
            static constexpr bool isAoS = isAoSIn || isAoSOut;
            static constexpr bool isStrided = isStridedIn || isStridedOut;

            static constexpr unsigned numDims = NumDimsIn::value;
            static constexpr bool isFwd = (autoDetectIsFwd && isComplexOut) || (!autoDetectIsFwd && T_IsFwd::value);
        };

        template<
            class T_Library,
            class T_FFT_Properties
            >
        struct FFT_Impl
        {
        private:
            using Library = typename bmpl::apply< T_Library, T_FFT_Properties >::type;
            static constexpr bool isInplace = T_FFT_Properties::isInplace;
            using Input = typename T_FFT_Properties::Input;
            using Output = typename T_FFT_Properties::Output;

            Library lib_;
        public:

            explicit FFT_Impl(Input& input, Output& output): lib_(input, output)
            {
                static_assert(!isInplace, "Must not be called for inplace transforms");
            }

            explicit FFT_Impl(Input& inOut): lib_(inOut)
            {
                static_assert(isInplace, "Must not be called for out-of-place transforms");
            }

            void operator()(Input& input, Output& output)
            {
                static_assert(!isInplace, "Must not be called for inplace transforms");
                policies::ReadData<Input>()(input);
                lib_(input, output);
                policies::WriteData<Output>()(output);
            }

            void operator()(Input& inout)
            {
                static_assert(isInplace, "Must not be called for out-of-place transforms");
                policies::ReadData<Input>()(inout);
                lib_(inout);
                policies::WriteData<Output>()(inout);
            }
        };

    }  // namespace detail

    /**
     * Assembles an FFT class that can be queried with the type member
     *
     * Usage:
     *      1) The constructor takes the container(s) and may modify the memory returned by the GetRawPtr policy
     *         Note: An implementation may not need the memory at all
     *      2) The ()-Operator is applied to the container(s) and calls the ReadData<Input> policy, than performs the transform
     *         from the memories returned by the GetRawPtr policies and calls the WriteData<Output> policy afterwards
     *
     * Parameters:
     * @param T_Library FFT Library to use
     * @param T_Input   Input container type
     * @param T_Output  Output container type (Can be an InplaceType for in-place transforms)
     * @param T_IsFwd   Whether to use forward or backward transform (should have a bool ::value member) or AutoDetect (True for C2C or R2C, False for C2R)
     * @param T_NumDums Number of dimensions, AutoDetect for using the NumDims trait on Input/Output containers
     */
    template<
            class T_Library,
            typename T_Input,
            typename T_Output = types::InplaceType<>,
            typename T_IsFwd = AutoDetect,
            typename T_NumDims = AutoDetect
            >
    struct FFT
    {
        using Library = T_Library;
        using Input = T_Input;
        using Output = T_Output;
        using IsFwd = T_IsFwd;
        using NumDims = T_NumDims;

        using FFT_Properties = detail::FFT_Properties< Input, Output, IsFwd, NumDims >;

        using type = detail::FFT_Impl< Library, FFT_Properties >;
    };
}  // namespace policies
}  // namespace foobar
