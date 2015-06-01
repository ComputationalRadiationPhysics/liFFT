#pragma once

#include <fftw3.h>
#include <cassert>
#include <boost/utility.hpp>
#include "foobar/policies/GetExtentsRawPtr.hpp"
#include "foobar/policies/GetRawPtr.hpp"
#include "foobar/libraries/fftw/policies/CreatePlan.hpp"
#include "foobar/libraries/fftw/policies/ExecutePlan.hpp"
#include "foobar/libraries/fftw/policies/FreePlan.hpp"
#include "foobar/libraries/fftw/policies/Ptr2ComplexPtr.hpp"

namespace foobar {
namespace libraries {
namespace fftw {

        namespace detail {

            template<
                    typename T_Precision,
                    bool T_isFwd,
                    bool T_isInplace,
                    unsigned T_numDims,
                    bool T_isComplexIn,
                    bool T_isComplexOut,
                    bool T_isAoSIn,
                    bool T_isAoSOut,
                    bool T_isStridedIn,
                    bool T_isStridedOut
                    >
            struct Planner;

            template< typename T_Precision, bool T_isFwd, bool T_isInplace, unsigned T_numDims >
            struct Planner< T_Precision, T_isFwd, T_isInplace, T_numDims, true, true, true, true, false, false >
            {
                using PlanType = typename traits::Types<T_Precision>::PlanType;

                template< class T_Input, class T_Output >
                typename std::enable_if< !T_isInplace, PlanType >::type
                operator()(const T_Input& input, T_Output& output, unsigned flags = FFTW_ESTIMATE)
                {
                    const unsigned* extends = foobar::policies::GetExtentsRawPtr< T_Input >(input)();
                    const unsigned* extendsOut = foobar::policies::GetExtentsRawPtr< T_Output >(output)();
                    for(unsigned i=0; i<T_numDims; ++i)
                        assert(extends[i] == extendsOut[i]);
                    T_Precision* inPtr = foobar::policies::GetRawPtr<T_Input>(input)();
                    T_Precision* outPtr = foobar::policies::GetRawPtr<T_Output>(output)();
                    return policies::CreatePlan<T_Precision>()(
                            T_numDims,
                            extends,
                            policies::FFTW_ptr2ComplexPtr(inPtr),
                            policies::FFTW_ptr2ComplexPtr(outPtr),
                            traits::Sign<T_isFwd>::value,
                            flags
                            );
                }

                template< class T_Input>
                typename std::enable_if< T_isInplace, PlanType >::type
                operator()(const T_Input& input)
                {
                    const unsigned* extends = foobar::policies::GetExtentsRawPtr< T_Input >(input)();
                    T_Precision* inPtr = foobar::policies::GetRawPtr<T_Input>(input)();
                    return policies::CreatePlan<T_Precision>()(
                            T_numDims,
                            extends,
                            policies::FFTW_ptr2ComplexPtr(inPtr),
                            policies::FFTW_ptr2ComplexPtr(inPtr),
                            traits::Sign<T_isFwd>::value,
                            FFTW_ESTIMATE
                            );
                }
            };

        }  // namespace detail

        template< class T_FFT >
        class FFTW: private boost::noncopyable
        {
        private:
            using FFT = T_FFT;
            using Planner =
                    detail::Planner<
                        FFT::PrecisionType,
                        FFT::isFwd,
                        FFT::isInplace,
                        FFT::numDims,
                        FFT::isComplexIn,
                        FFT::isComplexOut,
                        FFT::isAoSIn,
                        FFT::isAoSOut,
                        FFT::isStridedIn,
                        FFT::isStridedOut
                    >;
            using Executer = policies::ExecutePlan<FFT::PrecisionType>;
            using PlanDestroyer = policies::FreePlan<FFT::PrecisionType>;
            using PlanType = traits::Types<FFT::PrecisionType>::PlanType;

            typename PlanType plan_;

        public:
            template< typename std::enable_if< !FFT::isInplace >::type* = nullptr >
            FFTW(const FFT::Input& input, FFT::Output& output)
            {
                plan_ = Planner()(input, output);
            }

            template< typename std::enable_if< FFT::isInplace >::type* = nullptr >
            FFTW(const FFT::Input& input)
            {
                plan_ = Planner()(input);
            }

            ~FFTW()
            {
                PlanDestroyer()(plan_);
            }

            void operator()()
            {
                Executer()(plan_);
            }
        };

}  // namespace fftw
}  // namespace libraries
}  // namespace foobar
