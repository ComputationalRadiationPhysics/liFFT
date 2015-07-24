#pragma once

#include "foobar/FFT_Kind.hpp"
#include "foobar/AutoDetect.hpp"
#include "foobar/c++14_types.hpp"
#include "foobar/util.hpp"
#include "foobar/FFT_DataWrapper.hpp"
#include "foobar/FFT_InplaceOutput.hpp"

namespace foobar {

    namespace detail {

        template<class T_FFT_Def, bool T_isInplace = T_FFT_Def::isInplace>
        struct CreateFFT_Output
        {
            using FFT_Def = T_FFT_Def;
            static constexpr bool isComplexOutput = FFT_Def::isComplexOutput;
            static constexpr FFT_Kind kind = FFT_Def::kind;

            /**
             * The type of the output wrapper for a given input wrapper type
             * Use \ref getNewFFT_Output to get an instance of this!
             */
            template< typename T_Wrapper >
            using OutputWrapper = FFT_OutputDataWrapper<
                    FFT_Def,
                    std::conditional_t<
                        isComplexOutput,
                        mem::ComplexContainer< T_Wrapper::numDims, typename T_Wrapper::PrecisionType >,
                        mem::RealContainer< T_Wrapper::numDims, typename T_Wrapper::PrecisionType >
                    >,
                    std::true_type
                >;

            /**
             * Gets an instance of a FFT_OutputDataWrapper for the given
             * @param fftInput InputDataWrapper
             * @return Object to access the FFT output after an FFT
             */
            template< typename T_Wrapper >
            static OutputWrapper<T_Wrapper>
            getNewFFT_Output(const T_Wrapper& fftInput)
            {
                static_assert(kind != FFT_Kind::Complex2Real, "Missing argument for the full size of the last dimension");
                return getNewFFT_Output(fftInput, fftInput.getExtents()[T_Wrapper::numDims - 1]);
            }

            /**
             * Gets an instance of a FFT_OutputDataWrapper for the given
             * @param fftInput InputDataWrapper
             * @param fullSizeLastDim FullSize of the last dimension for C2R/R2C FFTs (optional for R2C)
             * @return Object to access the FFT output after an FFT
             */
            template< typename T_Wrapper >
            static OutputWrapper<T_Wrapper>
            getNewFFT_Output(const T_Wrapper& fftInput, unsigned fullSizeLastDim)
            {
                static_assert(std::is_same< typename T_Wrapper::FFT_Def, FFT_Def >::value, "Wrong wrapper passed in!");
                using Base = typename OutputWrapper<T_Wrapper>::Base;
                typename Base::IdxType extents;

                for(unsigned i=0; i<T_Wrapper::numDims; ++i)
                    extents[i] = fftInput.getExtents()[i];

                if(kind == FFT_Kind::Real2Complex)
                    extents[T_Wrapper::numDims - 1] = extents[T_Wrapper::numDims - 1]/2 + 1;
                else if(kind == FFT_Kind::Complex2Real)
                {
                    if(fullSizeLastDim/2 + 1 != extents[T_Wrapper::numDims - 1 ])
                        throw std::runtime_error("Wrong extents for last dim given");
                    extents[T_Wrapper::numDims - 1 ] = fullSizeLastDim;
                }
                return OutputWrapper<T_Wrapper>(Base(extents));
            }
        };

        template<class T_FFT_Def>
        struct CreateFFT_Output< T_FFT_Def, true >
        {
            using FFT_Def = T_FFT_Def;
            static constexpr bool isComplexOutput = FFT_Def::isComplexOutput;
            static constexpr FFT_Kind kind = FFT_Def::kind;

            /**
             * Gets an instance of a FFT_OutputDataWrapper for the given
             * @param fftInput InputDataWrapper
             * @return Object to access the FFT output after an FFT
             */
            template< typename T_Wrapper >
            static FFT_InplaceOutput<T_Wrapper>
            getNewFFT_Output(T_Wrapper& fftInput)
            {
                static_assert(kind != FFT_Kind::Complex2Real, "Missing argument for the full size of the last dimension");
                return getNewFFT_Output(fftInput, fftInput.getExtents()[T_Wrapper::numDims - 1]);
            }

            /**
             * Gets an instance of a FFT_OutputDataWrapper for the given
             * @param fftInput InputDataWrapper
             * @param fullSizeLastDim FullSize of the last dimension for C2R/R2C FFTs (optional for R2C)
             * @return Object to access the FFT output after an FFT
             */
            template< typename T_Wrapper >
            static FFT_InplaceOutput<T_Wrapper>
            getNewFFT_Output(T_Wrapper& fftInput, unsigned fullSizeLastDim)
            {
                static_assert(std::is_same< typename T_Wrapper::FFT_Def, FFT_Def >::value, "Wrong wrapper passed in!");

                return FFT_InplaceOutput<T_Wrapper>(fftInput);
            }
        };
    }  // namespace detail

    /**
     * Defines a FFT
     *
     * \tparam T_Library FFT-Library to use
     * \tparam T_kind Type of FFT to execute (Complex-Complex, Real-Complex...)
     * \tparam T_numDims Dimensionality of the FFT
     * \tparam T_PrecisionType Base type to use (float, double, ...)
     * \tparam T_IsFwd Whether to perform a forward FFT. Can be left at AutoDetect so it is true for Real-Complex and false for Complex-Real
     * \tparam T_isInplace Whether to perform the FFT inplace, that is, it does not allocate separate memory for the output
     */
    template<
        FFT_Kind T_kind,
        unsigned T_numDims,
        typename T_PrecisionType,
        class T_IsFwd = AutoDetect,
        bool T_isInplace = false
        >
    struct FFT_Definition
    {
        static constexpr FFT_Kind kind = T_kind;
        static constexpr unsigned numDims = T_numDims;
        using PrecisionType = T_PrecisionType;
        using IsFwd = T_IsFwd;
        static constexpr bool isInplace = T_isInplace;

        static constexpr bool autoDetectIsFwd = std::is_same< T_IsFwd, AutoDetect >::value;
        static_assert(
                !autoDetectIsFwd || kind!=FFT_Kind::Complex2Complex,
                "Auto-detect is not possible for Complex-Complex transforms");

        static constexpr bool isComplexInput = traits::IsComplexInput<kind>::value;
        static constexpr bool isComplexOutput = traits::IsComplexOutput<kind>::value;
        static constexpr bool isFwd = (autoDetectIsFwd && isComplexOutput) || (!autoDetectIsFwd && IsFwd::value);

        /**
         * Factory method to create a DataWrapper that can then be used as input for a FFT
         * If the accessor returns a non-const reference, the underlying data is assumed
         * to be continuous and directly used in the FFT
         * Otherwise temporary memory is created to which the data is copied before each FFT-invocation
         *
         * @param base Container holding the data
         * @param acc Accessor used to access the elements in the container via ()-operator(index, base)
         * @return DataWrapper that can then be used as input for a FFT and provides a default Accessor to access the underlying data
         */
        template<typename T_Base, typename T_BaseAccessor = traits::IdentityAccessor_t<T_Base> >
        static FFT_InputDataWrapper<
                FFT_Definition,
                std::remove_reference_t<T_Base>,
                negate< std::is_lvalue_reference<T_Base> >,
                std::remove_reference_t<T_BaseAccessor>
            >
        wrapFFT_Input(T_Base&& base, T_BaseAccessor&& acc = T_BaseAccessor())
        {
            return FFT_InputDataWrapper<
                        FFT_Definition,
                        std::remove_reference_t<T_Base>,
                        negate< std::is_lvalue_reference<T_Base> >,
                        std::remove_reference_t<T_BaseAccessor>
                    >(std::forward<T_Base>(base), std::forward<T_BaseAccessor>(acc));
        }

        /**
         * Factory method to create a DataWrapper that can then be used as output of a FFT
         * If the accessor returns a non-const reference, the underlying data is assumed
         * to be continuous and overwritten in the FFT
         * Otherwise temporary memory is created from which the data is copied after each FFT-invocation
         *
         * @param base Container holding the data
         * @param acc Accessor used to access the elements in the container via ()-operator(index, base)
         * @return DataWrapper that can then be used as input for a FFT and provides a default Accessor to access the underlying data
         */
        template< typename T_Base, typename T_BaseAccessor = traits::IdentityAccessor_t<T_Base> >
        static FFT_OutputDataWrapper<
                FFT_Definition,
                std::remove_reference_t<T_Base>,
                negate< std::is_lvalue_reference<T_Base> >,
                std::remove_reference_t<T_BaseAccessor>
            >
        wrapFFT_Output(T_Base&& base, T_BaseAccessor&& acc = T_BaseAccessor())
        {
            return FFT_OutputDataWrapper<
                        FFT_Definition,
                        std::remove_reference_t<T_Base>,
                        negate< std::is_lvalue_reference<T_Base> >,
                        std::remove_reference_t<T_BaseAccessor>
                    >(std::forward<T_Base>(base), std::forward<T_BaseAccessor>(acc));
        }

        using CreateFFT_Output = detail::CreateFFT_Output<FFT_Definition>;

        /**
         * Gets an instance of a FFT_OutputDataWrapper for the given
         * @param fftInput InputDataWrapper
         * @return Object to access the FFT output after an FFT
         */
        template< typename T_Wrapper >
        static auto
        getNewFFT_Output(T_Wrapper& fftInput)
        -> decltype(CreateFFT_Output::getNewFFT_Output(fftInput))
        {
            return CreateFFT_Output::getNewFFT_Output(fftInput);
        }

        /**
         * Gets an instance of a FFT_OutputDataWrapper for the given
         * @param fftInput InputDataWrapper
         * @param fullSizeLastDim FullSize of the last dimension for C2R/R2C FFTs (optional for R2C)
         * @return Object to access the FFT output after an FFT
         */
        template< typename T_Wrapper >
        static auto
        getNewFFT_Output(T_Wrapper& fftInput, unsigned fullSizeLastDim)
        -> decltype(CreateFFT_Output::getNewFFT_Output(fftInput, fullSizeLastDim))
        {
            return CreateFFT_Output::getNewFFT_Output(fftInput, fullSizeLastDim);
        }

    };

    // Some definitions for commonly used FFT types

    template< typename T_Precision = float, bool T_isInplace = false >
    using FFT_2D_C2C = FFT_Definition< FFT_Kind::Complex2Complex, 2, T_Precision, std::true_type, T_isInplace >;
    template< bool T_isInplace = false >
    using FFT_2D_C2C_F = FFT_2D_C2C< float, T_isInplace >;
    template< bool T_isInplace = false >
    using FFT_2D_C2C_D = FFT_2D_C2C< double, T_isInplace >;

    template< typename T_Precision = float, bool T_isInplace = false >
    using FFT_2D_R2C = FFT_Definition< FFT_Kind::Real2Complex, 2, T_Precision, AutoDetect, T_isInplace >;
    template< bool T_isInplace = false >
    using FFT_2D_R2C_F = FFT_2D_R2C< float, T_isInplace >;
    template< bool T_isInplace = false >
    using FFT_2D_R2C_D = FFT_2D_R2C< double, T_isInplace >;

    template< typename T_Precision = float, bool T_isInplace = false >
    using FFT_2D_C2R = FFT_Definition< FFT_Kind::Complex2Real, 2, T_Precision, AutoDetect, T_isInplace >;
    template< bool T_isInplace = false >
    using FFT_2D_C2R_F = FFT_2D_C2R< float, T_isInplace >;
    template< bool T_isInplace = false >
    using FFT_2D_C2R_D = FFT_2D_C2R< double, T_isInplace >;

    template< typename T_Precision = float, bool T_isInplace = false >
    using FFT_3D_C2C = FFT_Definition< FFT_Kind::Complex2Complex, 3, T_Precision, std::true_type, T_isInplace >;
    template< bool T_isInplace = false >
    using FFT_3D_C2C_F = FFT_3D_C2C< float, T_isInplace >;
    template< bool T_isInplace = false >
    using FFT_3D_C2C_D = FFT_3D_C2C< double, T_isInplace >;

    template< typename T_Precision = float, bool T_isInplace = false >
    using FFT_3D_R2C = FFT_Definition< FFT_Kind::Real2Complex, 3, T_Precision, AutoDetect, T_isInplace >;
    template< bool T_isInplace = false >
    using FFT_3D_R2C_F = FFT_3D_R2C< float, T_isInplace >;
    template< bool T_isInplace = false >
    using FFT_3D_R2C_D = FFT_3D_R2C< double, T_isInplace >;

    template< typename T_Precision = float, bool T_isInplace = false >
    using FFT_3D_C2R = FFT_Definition< FFT_Kind::Complex2Real, 3, T_Precision, AutoDetect, T_isInplace >;
    template< bool T_isInplace = false >
    using FFT_3D_C2R_F = FFT_3D_C2R< float, T_isInplace >;
    template< bool T_isInplace = false >
    using FFT_3D_C2R_D = FFT_3D_C2R< double, T_isInplace >;

}  // namespace foobar
