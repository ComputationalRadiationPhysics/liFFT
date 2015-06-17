#pragma once

#include <boost/utility.hpp>
#include <boost/mpl/placeholders.hpp>
#include "foobar/libraries/fftw/policies/Planner.hpp"
#include "foobar/libraries/fftw/policies/ExecutePlan.hpp"
#include "foobar/libraries/fftw/policies/FreePlan.hpp"

namespace bmpl = boost::mpl;

namespace foobar {
namespace libraries {
namespace fftw {

    /**
     * Wrapper for the CUDA-Library that executes the FFT on GPU(s)
     *
     * @param T_FFT_Properties Placeholder that will be replaced by a class containing the properties for this FFT
     */
    template< class T_FFT_Properties = bmpl::_1 >
    class FFTW: private boost::noncopyable
    {
    private:
        using FFT = T_FFT_Properties;
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
        PrecisionType *InPtr, *OutPtr;

    public:
        explicit FFTW(Input& input, Output& output)
        {
            plan_ = Planner()(input, output);
            InPtr = foobar::policies::GetRawPtr<Input>()(input);
            OutPtr = foobar::policies::GetRawPtr<Output>()(output);
        }

        explicit FFTW(Input& inOut)
        {
            plan_ = Planner()(inOut);
            InPtr = foobar::policies::GetRawPtr<Input>()(inOut);
            OutPtr = nullptr;
        }

        ~FFTW()
        {
            PlanDestroyer()(plan_);
        }

        void operator()(Input& input, Output& output)
        {
            if(foobar::policies::GetRawPtr<Input>()(input) != InPtr ||
                    foobar::policies::GetRawPtr<Output>()(output) != OutPtr)
                throw std::runtime_error("Pointers to data must not be changed after initialization");

            Executer()(plan_);
        }

        void operator()(Input& inOut)
        {
            if(foobar::policies::GetRawPtr<Input>()(inOut) != InPtr)
                throw std::runtime_error("Pointer to data must not be changed after initialization");
            Executer()(plan_);
        }
    };

}  // namespace fftw
}  // namespace libraries
}  // namespace foobar
