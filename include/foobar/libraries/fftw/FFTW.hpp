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
     * \tparam T_FFT_Properties Placeholder that will be replaced by a class containing the properties for this FFT
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
        using PlanType = typename traits::LibTypes< PrecisionType >::PlanType;
        using InPtr = std::result_of_t< decltype(&Input::getDataPtr)(Input) >;
        using OutPtr = std::result_of_t< decltype(&Output::getDataPtr)(Output) >;

        PlanType plan_;
        InPtr inPtr_;
        OutPtr outPtr_;

    public:
        FFTW(Input& input, Output& output)
        {
            plan_ = Planner()(input, output);
            inPtr_ = input.getDataPtr();
            outPtr_ = output.getDataPtr();
        }

        explicit FFTW(Input& inOut)
        {
            plan_ = Planner()(inOut);
            inPtr_ = inOut.getDataPtr();
            outPtr_ = nullptr;
        }

        FFTW(FFTW&& obj): plan_(std::move(obj.plan_)), inPtr_(obj.inPtr_), outPtr_(obj.outPtr_){}

        ~FFTW()
        {
            PlanDestroyer()(plan_);
        }

        void operator()(Input& input, Output& output)
        {
            if(input.getDataPtr() != inPtr_ || output.getDataPtr() != outPtr_)
                throw std::runtime_error("Pointers to data must not be changed after initialization");

            Executer()(plan_);
        }

        void operator()(Input& inOut)
        {
            if(inOut.getDataPtr() != inPtr_)
                throw std::runtime_error("Pointer to data must not be changed after initialization");
            Executer()(plan_);
        }
    };

}  // namespace fftw
}  // namespace libraries
}  // namespace foobar
