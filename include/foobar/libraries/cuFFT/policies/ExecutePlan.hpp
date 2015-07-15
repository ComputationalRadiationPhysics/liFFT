#pragma once

#include <cassert>
#include <type_traits>
#include <cufft.h>
#include "foobar/types/TypePair.hpp"
#include "foobar/libraries/cuFFT/Plan.hpp"
#include "foobar/libraries/cuFFT/traits/FFTType.hpp"
#include "foobar/libraries/cuFFT/traits/Sign.hpp"
#include "foobar/libraries/cuFFT/traits/LibTypes.hpp"
#include "foobar/policies/SafePtrCast.hpp"

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
          cufftResult operator()(cufftHandle plan, InType* in, OutType* out)
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
    struct ExecutePlan
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

        using LibTypes = traits::LibTypes< Precision, isComplexIn, isComplexOut >;
        using LibInType = typename LibTypes::InType;
        using LibOutType = typename LibTypes::OutType;

        using Executer = detail::ExecutePlan< Precision, isComplexIn, isComplexOut, isFwd >;

    public:
        template< class T_Plan, class T_Copier >
        void
        operator()(T_Plan& plan, Input& input, Output& output, const T_Copier& copy)
        {
            using foobar::policies::safe_ptr_cast;
            static_assert(!isInplace, "Cannot be used for inplace transforms!");

            auto pIn = safe_ptr_cast<LibInType*>(input.getDataPtr());
            if( plan.InDevicePtr )
            {
                unsigned numElements = input.getNumElements();
                copy.H2D(plan.InDevicePtr.get(), pIn, numElements * sizeof(LibInType));
                pIn = plan.InDevicePtr.get();
            }
            auto pOut = (plan.OutDevicePtr) ? plan.OutDevicePtr.get() : safe_ptr_cast<LibOutType*>(output.getDataPtr());
            cufftResult result = Executer()(plan.handle, pIn, pOut);
            if(result != CUFFT_SUCCESS)
                throw std::runtime_error("Error executing plan: " + std::to_string(result));
            if( plan.OutDevicePtr )
            {
                unsigned numElements = output.getNumElements();
                pOut = safe_ptr_cast<LibOutType*>(output.getDataPtr());
                copy.D2H(pOut, plan.OutDevicePtr.get(), numElements * sizeof(LibOutType));
            }
        }

        template< class T_Plan, class T_Copier >
        void
        operator()(T_Plan& plan, Input& inOut, const T_Copier& copy)
        {
            using foobar::policies::safe_ptr_cast;
            static_assert(isInplace, "Must be used for inplace transforms!");

            auto pIn = safe_ptr_cast<LibInType*>(inOut.getDataPtr());
            if( plan.InDevicePtr )
            {
                unsigned numElements = inOut.getNumElements();
                copy.H2D(plan.InDevicePtr.get(), pIn, numElements * sizeof(LibInType));
                pIn = plan.InDevicePtr.get();
            }
            cufftResult result = Executer()(plan.handle, pIn, pIn);
            if(result != CUFFT_SUCCESS)
                throw std::runtime_error("Error executing plan: " + std::to_string(result));
            if( plan.InDevicePtr )
            {
                unsigned numElements = inOut.getNumElements();
                auto pOut = safe_ptr_cast<LibOutType*>(inOut.getDataPtr());
                copy.D2H(pOut, plan.InDevicePtr.get(), numElements * sizeof(LibOutType));
            }
        }
    };

}  // namespace policies
}  // namespace cuFFT
}  // namespace libraries
}  // namespace foobar
