#pragma once

#include <cassert>
#include <type_traits>
#include <cufft.h>
#include "foobar/types/TypePair.hpp"
#include "foobar/policies/GetRawPtr.hpp"
#include "foobar/libraries/cuFFT/Plan.hpp"
#include "foobar/libraries/cuFFT/traits/FFTType.hpp"
#include "foobar/libraries/cuFFT/traits/Sign.hpp"
#include "foobar/libraries/cuFFT/traits/Types.hpp"
#include "foobar/libraries/cuFFT/traits/LibTypes.hpp"

namespace foobar {
namespace libraries {
namespace cuFFT {

namespace policies {

    namespace detail {

        template<
                typename T_Precision,
                bool T_isComplexIn,
                bool T_isComplexOut,
                bool T_isFwd
        >
        struct ExecutePlan;

        template< bool T_isFwd >
        struct ExecutePlan< float, true, true, T_isFwd >: traits::LibTypes< float, true, true >
        {
          cufftResult operator()(cufftHandle plan, InType* in, OutType* out)
          {
              return cufftExecC2C(plan, in, out, traits::Sign< T_isFwd >::value);
          }
        };

        template< bool T_isFwd >
        struct ExecutePlan< float, false, true, T_isFwd >: traits::LibTypes< float, false, true >
        {
          cufftResult operator()(cufftHandle plan, InType* in, OutType* out)
          {
              static_assert(T_isFwd, "R2C is always a forward transform!");
              return cufftExecR2C(plan, in, out);
          }
        };

        template< bool T_isFwd >
        struct ExecutePlan< float, true, false, T_isFwd >: traits::LibTypes< float, true, false >
        {
          cufftResult operator()(cufftHandle plan, typename InType* in, OutType* out)
          {
              static_assert(T_isFwd, "C2R is always a inverse transform!");
              return cufftExecC2R(plan, in, out);
          }
        };

        template< bool T_isFwd >
        struct ExecutePlan< double, true, true, T_isFwd >: traits::LibTypes< double, true, true >
        {
          cufftResult operator()(cufftHandle plan, InType* in, OutType* out)
          {
              return cufftExecZ2Z(plan, in, out, traits::Sign< T_isFwd >::value);
          }
        };

        template< bool T_isFwd >
        struct ExecutePlan< double, false, true, T_isFwd >: traits::LibTypes< double, false, true >
        {
          cufftResult operator()(cufftHandle plan, InType* in, OutType* out)
          {
              static_assert(T_isFwd, "R2C is always a forward transform!");
              return cufftExecD2Z(plan, in, out);
          }
        };

        template< bool T_isFwd >
        struct ExecutePlan< double, true, false, T_isFwd >: traits::LibTypes< double, true, false >
        {
          cufftResult operator()(cufftHandle plan, InType* in, OutType* out)
          {
              static_assert(T_isFwd, "C2R is always a inverse transform!");
              return cufftExecZ2D(plan, in, out);
          }
        };

    }  // namespace detail

    template<
            typename T_Precision,
            class T_InOutTypes,
            bool T_isFwd,
            bool T_isInplace,
            unsigned T_numDims,
            bool T_isComplexIn,
            bool T_isComplexOut
    >
    struct ExecutePlan: traits::LibTypes< T_Precision, T_isComplexIn, T_isComplexOut >
    {
    private:
        using Precision = T_Precision;
        using Input = typename T_InOutTypes::First;
        using Output = typename T_InOutTypes::Second;
        static constexpr bool isFwd = T_isFwd;
        static constexpr unsigned numDims = T_numDims;
        static constexpr bool isInplace = T_isInplace;
        static constexpr bool  isComplexIn = T_isComplexIn;
        static constexpr bool  isComplexOut = T_isComplexOut;

        using RawPtrIn = foobar::policies::GetRawPtr< Input >;
        using RawPtrOut = foobar::policies::GetRawPtr< Output >;
        using PtrConverterIn = foobar::policies::Ptr2Ptr< Precision, LibInType >;
        using PtrConverterOut = foobar::policies::Ptr2Ptr< Precision, LibOutType >;
        using Executer = detail::ExecutePlan< Precision, isComplexIn, isComplexOut, isFwd >;

    public:
        using PlanType = Plan< LibInType, LibOutType >;

        template< class T_Copier >
        void
        operator()(PlanType& plan, Input& input, Output& output, T_Copier& copy)
        {
            static_assert(!isInplace, "Cannot be used for inplace transforms!");

            RawPtrIn ptrIn;
            RawPtrOut ptrOut;
            auto pIn = PtrConverterIn()(ptrIn(input));
            if( plan.InDevicePtr != nullptr)
            {
                unsigned numElements = foobar::policies::GetNumElements< Input, numDims >()(input);
                copy.H2D( plan.InDevicePtr, pIn, numElements * sizeof(LibInType));
                pIn = plan.InDevicePtr;
            }
            auto pOut = (plan.OutDevicePtr) ? plan.OutDevicePtr : PtrConverterOut()(ptrOut(output));
            cufftResult result = Executer()(plan.plan, pIn, pOut);
            if(result != CUFFT_SUCCESS)
                throw std::runtime_error("Error executing plan: " + std::to_string(result));
            if( plan.OutDevicePtr != nullptr)
            {
                unsigned numElements = foobar::policies::GetNumElements< Output, numDims >()(output);
                pOut = PtrConverterOut()(ptrOut(output));
                copy.D2H(pOut, plan.OutDevicePtr, numElements * sizeof(LibInType));
            }
        }

        template< class T_Copier >
        void
        operator()(PlanType& plan, Input& inOut, T_Copier& copy)
        {
            static_assert(isInplace, "Must be used for inplace transforms!");

            RawPtrIn ptrIn;
            auto pIn = PtrConverterIn()(ptrIn(inOut));
            if( plan.InDevicePtr != nullptr)
            {
                unsigned numElements = foobar::policies::GetNumElements< Input, numDims >()(inOut);
                copy.H2D( plan.InDevicePtr, pIn, numElements * sizeof(LibInType));
                pIn = plan.InDevicePtr;
            }
            cufftResult result = Executer()(plan.plan, pIn, pIn);
            if(result != CUFFT_SUCCESS)
                throw std::runtime_error("Error executing plan: " + std::to_string(result));
            if( plan.InDevicePtr != nullptr)
            {
                unsigned numElements = foobar::policies::GetNumElements< Output, numDims >()(inOut);
                pIn = PtrConverterOut()(ptrIn(inOut));
                copy.H2D(pIn, plan.InDevicePtr, numElements * sizeof(LibInType));
            }
        }
    };

}  // namespace policies
}  // namespace cuFFT
}  // namespace libraries
}  // namespace foobar
