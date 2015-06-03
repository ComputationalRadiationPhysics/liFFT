#pragma once

#include <boost/utility.hpp>
#include "foobar/libraries/fftw/policies/Planner.hpp"
#include "foobar/libraries/fftw/policies/ExecutePlan.hpp"
#include "foobar/libraries/fftw/policies/FreePlan.hpp"

namespace foobar {
namespace libraries {
namespace fftw {

    template< class T_FFT >
    class FFTW: private boost::noncopyable
    {
    private:
        using FFT = T_FFT;
        using Input = typename FFT::Input;
        using Output = typename FFT::Output;
        using PrecisionType = typename FFT::PrecisionType;
        using Planner =
                policies::Planner<
                    PrecisionType,
                    foobar::types::TypePair< Input, Output >,
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
        using Executer = policies::ExecutePlan< PrecisionType >;
        using PlanDestroyer = policies::FreePlan< PrecisionType >;
        using PlanType = typename traits::Types< PrecisionType >::PlanType;

        PlanType plan_;

    public:
        explicit FFTW(const Input& input, Output& output)
        {
            plan_ = Planner()(input, output);
        }

        explicit FFTW(const Input& inOut)
        {
            plan_ = Planner()(inOut);
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
