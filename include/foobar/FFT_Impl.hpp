#pragma once

#include <boost/mpl/apply.hpp>
#include "foobar/policies/ReadData.hpp"
#include "foobar/policies/WriteData.hpp"

namespace bmpl = boost::mpl;

namespace foobar {
namespace detail {

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
}  // namespace foobar
